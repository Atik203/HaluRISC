# HaluRISC — Detailed Implementation Roadmap

**Goal:** A defensible Version A course project, extended on `version-B` into a publication study (leakage control, cross-domain robustness, calibration under shift, explanation reliability, and a research web artifact). Heavy experiments run in Colab Pro; the local RTX 3060 Laptop GPU (6 GB VRAM) supports inference, profiling, UI, and demos.

**Convention:** items marked `[verified 2026]` were checked against current web/PyPI info in July 2026.

> **IMPLEMENTATION STATUS (updated 2026-08-10):** ✅ **~90% done.**
>
> Everything code-side is complete and verified: Version A integrity repair (leakage-free split, corrected artifacts, XGBoost F1 0.9842 / AUROC 0.9982) and all of Version B — B1 unified data, B2 baselines, B3 cross-domain, B4 calibration under shift, B5 explanation reliability, B6 reproducibility, B7 research UI, B7.5 conversational tiers 1–4 (206 tests, build + lint green, `ALL ARTIFACTS VERIFIED`).
>
> **What still needs to be done is human work — the manual review sheets and the paper — listed in §17 at the end of this file.**

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

> **Note:** large raw/processed datasets, model files, figures, and result artifacts remain gitignored and must be regenerated or restored from the Colab export.

```
HaluRISC/
├── blueprint.md            # research blueprint (source of truth)
├── proposal.md             # supervisor proposal (Markdown copy)
├── roadmap.md              # this file
├── report/                 # LaTeX proposal
├── docs/                   # phase guides: b5 (manual review), b6 (reproducibility), b7 (UI), frozen manifest
├── configs/                # version_b.yaml (run_all protocol)
├── data/
│   ├── raw/                # gitignored: HaluEval, official RAGTruth, FaithBench, halueval/
│   └── processed/          # cleaned parquet, feature matrix, unified records, retrieval index
├── src/
│   ├── data/               # download + prep + unified schema/registry
│   ├── features/           # extract_features.py + per-group modules
│   ├── claims/             # decompose.py, verify.py, judge.py (tiers 2/4)
│   ├── retrieval/          # chunk.py, index.py, web_search.py, rerank.py (tier 3)
│   ├── models/             # run_b2/b3/b4/b5, run_all, make_manifest, verify_artifacts, tune_thresholds, eval_claims, review_tally
│   ├── explain/            # shap_analysis.py
│   └── api/                # FastAPI app (main.py)
├── web/                    # Next.js app: /chat /analyze /dashboard/* /demo /about
├── colab/                  # self-contained HaluRISC_Training_Version_B.ipynb + cache helpers
├── artifacts/              # models/, results/ (b2-b5), figures/, split files
└── requirements.txt        # pinned versions (xgboost 3.3.0 exception)
```

---

## ✅ DONE — Version A (Phases 0–8)

Every phase below is complete and its outputs are verified. See `docs/` and the `artifacts/` paths for details.

### 3. Phase 0 — Environment Setup — ✅ DONE
`.venv` (Python 3.12) with pinned `requirements.txt`; pnpm workspace with Next 16 + Tailwind v4 + assistant-ui + AI SDK v7.

### 4. Phase 1 — Data Acquisition — ✅ DONE
All datasets downloaded and verified: HaluEval (20K rows), official RAGTruth (response/source files + revision hashes), FaithBench; 50-sample manual audit in `data/processed/audit_50_samples.json`.

### 5. Phase 2 — Preprocessing & Splits — ✅ DONE
Leakage-free grouped split by `item_idx` (70/15/15); split indices + integrity report in `artifacts/`.

### 6. Phase 3 — Feature Extraction — ✅ DONE
All 26 features across the 7 groups extracted once to `data/processed/features_full.parquet` (hash-verified against the frozen Colab run).

### 7. Phase 4 — Modeling — ✅ DONE
Heuristic / LR / RF / tuned XGBoost with grouped 5-fold CV, seeds 42/123/456; models in `artifacts/models/` (XGBoost F1 0.9842 / AUROC 0.9982).

### 8. Phase 5 — Calibration & Evaluation — ✅ DONE (evidence produced)
Platt/isotonic (fit on validation only), all metrics, statistics (McNemar, bootstrap CIs, Wilcoxon), ablation, RAGTruth zero-shot external check. Only the manual error-case review sheet is pending (see §17).

### 9. Phase 6 — Explainability — ✅ DONE
SHAP global + local figures, and its reliability is now measured in B5 (importance triangulation, neutralization, perturbation stability, bootstrap CIs).

### 10. Phase 7 — Backend API — ✅ DONE
FastAPI live and verified end-to-end: `/health /predict /explain /judge /meta /verify /index /feedback`, serving the B-run deployable (B2 XGBoost + B4 Platt). One worker, no `--reload`, inference lock, bounded inputs, LRU feature cache.

### 11. Phase 8 — Frontend Dashboard — ✅ DONE
Chat with auto risk cards, Analyze with compare mode, the 6-tab experiment dashboard, the offline `/demo` walkthrough, mobile nav and accessibility. `pnpm build` + `lint` pass. Runbook: `docs/b7-research-ui.md`.

---

## ✅ DONE — Version B (B0–B7.5)

### B0 — Version A integrity gate — ✅ DONE
Corrected grouped-split rerun complete, split report leakage-free, every artifact loads and predicts on this machine (`verify_artifacts.py` passes).

### B1 — Unified data schema — ✅ DONE (2026-08-06)
Canonical schema, label mappings, dataset registry/license manifest, official RAGTruth + FaithBench downloaders, `prepare_unified.py` → 38,540 deterministic rows; mapping + license reports in `artifacts/results/`.

### B2 — Corrected baseline and artifact controls — ✅ DONE (2026-08-06)
Nine baselines with grouped-CV tuning, leakage-removal impact report, per-seed predictions; XGBoost F1 0.9857 / AUROC 0.9980. Artifacts: `artifacts/results/b2/` + `artifacts/models/b2/`.

### B3 — Cross-domain robustness — ✅ DONE (2026-08-09 Colab run)
Zero-shot evaluation on RAGTruth + FaithBench finished with clean predictions (55,620 rows, no duplicates); honest transfer-gap numbers on the dashboard. Nothing pending.

### B4 — Calibration under distribution shift — ✅ DONE (2026-08-09)
Source/target calibration complete with correct counts (re-ran locally); headline on the dashboard: ECE 0.81 → 0.13 after target calibration.

### B5 — Explanation reliability and error analysis — ✅ DONE (2026-08-09)
Importance triangulation, neutralization, perturbations, stability, review export all done. **Only remaining step: the manual two-reviewer sheet** (guide: `docs/b5-explanation-reliability.md`; tally: `review_tally.py`) — see §17.

### B6 — Reproducible publication artifact — ✅ DONE (2026-08-09)
`run_all_experiments.py` (config-driven protocol), enriched frozen manifest (raw hashes, seeds, b5 entries, source fingerprint), CPU Dockerfile — all tested. Guide: `docs/b6-reproducibility.md`.

### B7 — Research UI and demo — ✅ DONE (2026-08-09)
Six-tab evidence dashboard, Analyze compare mode, offline `/demo` walkthrough, mobile nav + accessibility, verified (build + lint green). Guide: `docs/b7-research-ui.md`.

### B7.5 — Conversational auto-analysis (Tiers 1–4) — ✅ DONE (2026-08-09/10)
All four tiers live: auto risk cards per answer (T1), per-claim NLI verdicts with "Evidence says" quotes (T2), Tavily web + document retrieval with citations (T3), LLM-judge routing + feedback loop + threshold tuning + rate limits (T4). 206 tests pass.

---

### Colab notebook note

`colab/HaluRISC_Training_Version_B.ipynb` is the single self-contained notebook to use (cell 3 writes + hash-verifies the source; no zip upload). After any source change, regenerate it with `python colab/build_self_contained.py` and re-upload.

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
| Calibrated score misleads users on style-sensitivity | Card headline derives from per-claim evidence verdicts; legacy score shown as a labelled secondary signal |

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

---

## 17. What Still Needs to Be Done (all manual / human work)

Everything left is paper and review work — no code changes are required. Guides and artifacts are ready in `docs/`.

### 17.1 Manual review tasks (do these first)

1. **B5.5 two-reviewer sheet** — fill `reviewer_1` / `reviewer_2` / `agreement` for the ~40 cases in `artifacts/results/b5/b5_review_cases.csv`, save as `b5_review_cases_reviewed.csv`, then tally with `python src/models/review_tally.py`. Step-by-step guide: `docs/b5-explanation-reliability.md`.
2. **Claim-eval labeling (T2/T3 evals)** — build the human-labeling sheet with `python src/models/eval_claims.py --build`, label claims by hand, then measure verdict agreement and flagging precision/recall with `--evaluate`. For retrieval, annotate `--tier3` queries and report citation recall@k.
3. **Feedback labeling** — if you used the 👍/👎 buttons in chat, export the collected rows with `--from-feedback` and label `correct_verdict`; then `python src/models/tune_thresholds.py` to see whether tuned NLI thresholds improve agreement (`--apply` is an explicit human step).

### 17.2 B8 — Manuscript & delivery (paper work only)

1. Freeze the artifact manifest (already generated — `artifacts/results/manifest.json` and `docs/manifest.frozen.json`).
2. Write the paper around the five storylines: leakage control (B2), cross-domain robustness (B3), calibration under shift (B4), explanation reliability (B5), and deployment cost (B7 efficiency).
3. Include dataset licenses, limitations, source-group rules, and the negative results (the transfer gap and the style-sensitive calibrated score are honest negatives, not failures).
4. Do not claim SOTA, universal truth detection, guaranteed Q2 acceptance, or an unverified first contribution.
5. Prepare the 5-minute demo script: problem → grounded example → unsupported example → explanation → calibration/shift result → failure case → efficiency. The offline `/demo` page already contains the material.
6. Select the journal only after the corrected results are available; recheck scope, quartile, APC, and author guidelines at submission time.

Exit condition: manuscript numbers, dashboard numbers, manifest, and screenshots all come from the same frozen commit and artifact bundle.

### 17.3 Final checks

1. **Citation verification (before submission)** — click through every DOI/URL in the paper; the 2026 entries (`ijert`, `ieeeTai`, `multimedia`, `spikescore`) in particular must resolve.
2. **Paper mapping** — use the table below to pull each section's numbers from artifacts:

| Paper section | Artifact |
|---|---|
| Dataset | `data/processed/*.parquet` + audit + dataset/license manifest |
| Features | `src/features/` + feature table |
| Experiments | `artifacts/results/b2/` (3 seeds, grouped split, shortcut controls) |
| Cross-domain | `artifacts/results/b3/` (metrics, transfer, subgroups, CIs) |
| Calibration | `artifacts/results/b4/` (ECE/Brier/NLL, target story, reliability) |
| Explainability | `artifacts/results/b5/` + SHAP figures |
| Efficiency | latency/cost tables + LLM-judge comparison |
| Reproducibility | pinned requirements, split hashes, `run_all` protocol, frozen manifest, CPU Dockerfile |

3. **Verify live claims** — XGBoost `3.3.0` and `cross-encoder/nli-deberta-v3-base` model card still match before install (links in §16).
