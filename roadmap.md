# HaluRISC — Detailed Implementation Roadmap

**Goal:** A defensible Version A course project, extended on `version-B` into a publication study (leakage control, cross-domain robustness, calibration under shift, explanation reliability, and a research web artifact). Heavy experiments run in Colab Pro; the local RTX 3060 Laptop GPU (6 GB VRAM) supports inference, profiling, UI, and demos.

**Convention:** items marked `[verified 2026]` were checked against current web/PyPI info in July 2026.

> **IMPLEMENTATION STATUS (updated 2026-09-17): ✅ COMPLETE.**
>
> Both phases are finished and verified end to end. Version A: leakage-free grouped split, corrected artifacts (XGBoost F1 0.9842 / AUROC 0.9982), API + tests + portable Colab notebook. Version B: B1 unified data, B2 baselines, B3 cross-domain, B4 calibration under shift, B5 explanation reliability, B6 reproducibility, B7 research UI, B7.5 conversational tiers 1–4, B7.6 evidence-domain display score (206 tests, build + lint green, `ALL ARTIFACTS VERIFIED`).
>
> **Manuscripts are complete in two formats.** `report/paper.tex` (course format, 15-page clean build, hyperlinked citations, all Version B numbers) with `report/proposal.tex`, and `Journal_Paper/halurisc.tex` (Elsevier CAS single-column, 14-page clean build, six-section journal skeleton, 25 verified references). The API serves the evidence-domain display score (B4 natural isotonic + per-claim adjustment) instead of the blanket-99% HaluEval-Platt score.
>
> **Later additions (2026-09-17):** the B5.5 expert audit was completed (AI-assisted two passes, 23/26 agreement; sheet and tally in `artifacts/results/b5/`), the HaluEval answer-length confound was quantified (`b5_length_error_analysis.csv`) and folded into the journal manuscript, and HaluEval provenance was pinned (commit + SHA-256 in `revision.json` and the license manifest). `blueprint.md` is now a single unified design document.
>
> **What still needs to be done is human work only — audit review, claim/feedback labeling, and submission checks — listed in §17 at the end of this file.**

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
├── report/                 # course-format manuscript (paper.tex) + proposal + figures/screenshots
├── Journal_Paper/          # journal-format manuscript (halurisc.tex, CAS single column) + ref.bib + class files
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
FastAPI live and verified end-to-end: `/health /predict /explain /judge /meta /verify /index /feedback`, serving the B-run deployable (B2 XGBoost + B4 display isotonic fitted on natural RAGTruth data; HaluEval-Platt kept as `legacy_score`). One worker, no `--reload`, inference lock, bounded inputs, LRU feature cache.

### 11. Phase 8 — Frontend Dashboard — ✅ DONE
Chat with auto risk cards, Analyze with compare mode, the 6-tab experiment dashboard, the offline `/demo` walkthrough, mobile nav and accessibility. `pnpm build` + `lint` pass. Runbook: `docs/b7-research-ui.md`.

---

## ✅ DONE — Version B (B0–B7.5)

### B0 — Version A integrity gate — ✅ DONE
Corrected grouped-split rerun complete, split report leakage-free, every artifact loads and predicts on this machine (`verify_artifacts.py` passes).

### B1 — Unified data schema — ✅ DONE (2026-08-06)
Canonical schema, label mappings, dataset registry/license manifest, official RAGTruth + FaithBench downloaders, `prepare_unified.py` → 38,540 deterministic rows; mapping + license reports in `artifacts/results/`.

### B2 — Corrected baseline and artifact controls — ✅ DONE (2026-08-06)
Nine baselines with grouped-CV tuning, leakage-removal impact report, per-seed predictions; XGBoost F1 0.9846 / AUROC 0.9979. Artifacts: `artifacts/results/b2/` + `artifacts/models/b2/`.

### B3 — Cross-domain robustness — ✅ DONE (2026-08-09 Colab run)
Zero-shot evaluation on RAGTruth + FaithBench finished with clean predictions (55,620 rows, no duplicates); honest transfer-gap numbers on the dashboard. Nothing pending.

### B4 — Calibration under distribution shift — ✅ DONE (2026-08-09)
Source/target calibration complete with correct counts (re-ran locally); headline on the dashboard: ECE 0.81 → 0.13 after target calibration.

### B5 — Explanation reliability and error analysis — ✅ DONE (2026-08-09)
Importance triangulation, neutralization, perturbations, stability, and the review export are done. The B5.5 sheet was completed with two AI-assisted expert passes (23/26 agreement, 15 cases implausible on both passes), tallied with `review_tally.py`, and documented in `docs/b5-explanation-reliability.md`. Follow-up: the answer-length confound was quantified by `src/models/analyze_length_shortcut.py` → `b5_length_error_analysis.csv` and added to the journal Results.

### B6 — Reproducible publication artifact — ✅ DONE (2026-08-09)
`run_all_experiments.py` (config-driven protocol), enriched frozen manifest (raw hashes, seeds, b5 entries, source fingerprint), CPU Dockerfile — all tested. Guide: `docs/b6-reproducibility.md`.

### B7 — Research UI and demo — ✅ DONE (2026-08-09)
Six-tab evidence dashboard, Analyze compare mode, offline `/demo` walkthrough, mobile nav + accessibility, verified (build + lint green). Guide: `docs/b7-research-ui.md`.

### B7.5 — Conversational auto-analysis (Tiers 1–4) — ✅ DONE (2026-08-09/10)
All four tiers live: auto risk cards per answer (T1), per-claim NLI verdicts with "Evidence says" quotes (T2), Tavily web + document retrieval with citations (T3), LLM-judge routing + feedback loop + threshold tuning + rate limits (T4). 206 tests pass.

### B7.6 — Evidence-domain display score — ✅ DONE (2026-08-10)
`calibrated_score` no longer saturates at 99% on full-sentence inputs. `src/models/fit_display_calibrator.py` fits the B4 display calibrator (isotonic) on 5,034 natural RAGTruth QA rows (ECE 0.133 on the disjoint 900-row test, vs 0.819 raw); `/verify` further adjusts the score from per-claim verdicts (contradicted up, supported down). Verified live: grounded full-sentence answer 0.31 (low), contradicted-claim answer 0.80 (high).

### B8 — Manuscript & delivery — ✅ DONE (2026-08-10/11)
`report/paper.tex` rewritten as a modular document (intro, literature review, methodology, experimental setup, results, system, discussion, conclusion, reproducibility) with only verified Version B numbers, student-register prose (`report/paper_prompt.md`), hyperlinked citations, and vendored figures (pipeline infographic, system architecture, reliability/calibration/transfer diagrams). `report/proposal.tex` and `proposal.pdf` updated to Version B. All five 2026 citations verified against live sources and fixed in `ref.bib` (Luna authors, IJERT authors, IEEE TAI authors, Multimedia authors, SpikeScore ICLR). Later addition: the journal-format manuscript `Journal_Paper/halurisc.tex` (Elsevier CAS single-column, six-section skeleton, 25 references verified 2026-09-17, 14-page clean build) with the length-confound table and the audit paragraph.

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

The manuscript is written; everything left is manual review and final pre-submission checks — no code changes are required. Guides and artifacts are ready in `docs/`.

### 17.1 Manual review tasks

1. ~~B5.5 two-reviewer sheet~~ ✅ **DONE (2026-09-17, AI-assisted).** `artifacts/results/b5/b5_review_cases_reviewed.csv` holds the two passes (23/26 agreement, `review_mode = ai-expert`); tally with `python src/models/review_tally.py`. If the study ever needs a human two-reviewer result, the sheet must be re-filled by humans. Guide: `docs/b5-explanation-reliability.md`.
2. **Claim-eval labeling (T2/T3 evals)** — build the human-labeling sheet with `python src/models/eval_claims.py --build`, label claims by hand, then measure verdict agreement and flagging precision/recall with `--evaluate`. For retrieval, annotate `--tier3` queries and report citation recall@k.
3. **Feedback labeling** — if you used the 👍/👎 buttons in chat, export the collected rows with `--from-feedback` and label `correct_verdict`; then `python src/models/tune_thresholds.py` to see whether tuned NLI thresholds improve agreement (`--apply` is an explicit human step).
4. **50-sample audit human review** — `data/processed/audit_50_samples.json` exists but the human label-quality review fields are still pending.

### 17.2 Pre-submission delivery (paper work only)

1. The B5.5 sheet is filled (17.1). Claim-eval and feedback labels remain optional inputs if those numbers should appear in the paper.
2. Click through every DOI/URL at submission time: `report/ref.bib` (18 entries, verified 2026-08-11; `ieeeTai` DOI must be re-checked on ieeexplore, which blocks scraping) and `Journal_Paper/ref.bib` (25 entries, verified 2026-09-17).
3. If desired, update the System chapter screenshots (`report/screenshots/*.png`) to the latest chat auto-risk card, then rebuild both manuscripts: `latexmk -pdf -outdir=out paper.tex` (from `report/`) and `latexmk -pdf -outdir=out halurisc.tex` (from `Journal_Paper/`).
4. Prepare the 5-minute demo script: problem → grounded example → unsupported example → explanation → calibration/shift result → failure case → efficiency. The offline `/demo` page already contains the material.
5. Pick the journal venue and recheck scope, quartile, APC, and author guidelines at submission time. The CAS single-column manuscript in `Journal_Paper/` is the submission draft.

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
| Explanation audit + length confound | `b5_review_cases_reviewed.csv` + `b5_length_error_analysis.csv` |
| Efficiency | latency/cost tables + LLM-judge comparison |
| Reproducibility | pinned requirements, split hashes, `run_all` protocol, frozen manifest, CPU Dockerfile |

3. **Journal manuscript** — every number in `Journal_Paper/halurisc.tex` must trace to the same artifacts; the six-section skeleton follows the CAS sample, and `Conventional Method` is the journal name for the literature chapter (the course manuscript keeps `Literature Review`).
4. **Verify live claims** — XGBoost `3.3.0` and `cross-encoder/nli-deberta-v3-base` model card still match before install (links in §16).
