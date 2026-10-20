# HaluRISC — Detailed Implementation Roadmap

**Goal:** A defensible Version A course project, extended on `version-B` into a publication study (leakage control, cross-domain robustness, calibration under shift, explanation reliability, and a research web artifact). Heavy experiments run in Colab Pro; the local RTX 3060 Laptop GPU (6 GB VRAM) supports inference, profiling, UI, and demos.

**Convention:** items marked `[verified 2026]` were checked against current web/PyPI info in July 2026.

> **IMPLEMENTATION STATUS (updated 2026-10-20): ✅ COMPLETE.**
>
> Both phases are finished and verified end to end. Version A: leakage-free grouped split, corrected artifacts (XGBoost F1 0.9842 / AUROC 0.9982), API + tests + portable Colab notebook. Version B: B1 unified data, B2 baselines, B3 cross-domain, B4 calibration under shift, B5 explanation reliability, B6 reproducibility, B7 research UI, B7.5 conversational tiers 1–4, B7.6 evidence-domain display score (231 tests, build + lint green, `ALL ARTIFACTS VERIFIED`).
>
> **Manuscript is complete.** `Journal_Paper/halurisc.tex` (Elsevier CAS single-column, 17-page clean build, six-section journal skeleton, 38 verified references) is the submission draft, titled around EC-XGB (Evidence-Consistent XGBoost). The earlier course-format manuscript (`report/paper.tex` with `report/proposal.tex`) was retired in September 2026; its sources remain in git history. The API serves EC-XGB (`b6-ec-xgb-v1.0`) with its Platt display calibrator fitted on RAGTruth QA plus per-claim adjustment, and the claim splitter is atomic and conjunction-aware (``and``/``but`` clauses become separate claims).
>
> **Later additions (2026-09-17):** the B5.5 expert audit was completed (AI-assisted two passes, 23/26 agreement; sheet and tally in `artifacts/results/b5/`), the HaluEval answer-length confound was quantified (`b5_length_error_analysis.csv`) and folded into the journal manuscript, and HaluEval provenance was pinned (commit + SHA-256 in `revision.json` and the license manifest). `blueprint.md` is now a single unified design document.
>
> **What still needs to be done is human work only — audit review, claim/feedback labeling, and submission checks — listed in §17 at the end of this file.**
>
> **2026-10-20:** a new external pre-submission review (major revision) is now the active plan in §18. It supersedes §17 for the next cycle. The review's critical items are the calibration/threshold protocol, the length-shortcut logic, model-specific ECE labels, and zero-shot wording, not new baselines.

### Operating rule

1. Fix Version A first.
2. Run heavy training and evaluation in Colab Pro.
3. Download the corrected artifacts to the repository root.
4. Verify the API and UI locally on the RTX 3060 laptop.
5. Push the final Version A correction to `version-A`.
6. Only then implement Version B on `version-B`.

> For the §18 revision cycle (2026-10-20 onward), all new work runs locally. The 16 GB GPU handles the neural fine-tune; the 6 GB RTX 3060 handles the rest. Colab is a fallback only.

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
├── Journal_Paper/          # journal manuscript (halurisc.tex, CAS single column) + ref.bib + class files + figures/screenshots
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
FastAPI live and verified end-to-end: `/health /predict /explain /judge /meta /verify /index /feedback /predict/compare /answer`, serving EC-XGB (B9; artifact namespace `b6/`) with its own Platt display calibrator fitted on the RAGTruth QA calibration split. The standard B2 XGBoost stays loaded as `legacy_score` and as a comparison baseline; the other B2 baselines appear in `/predict/compare`. One worker, no `--reload`, inference lock, bounded inputs, LRU feature cache.

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
All four tiers live: auto risk cards per answer (T1), per-claim NLI verdicts with "Evidence says" quotes (T2), Brave (LLM Context) web + document retrieval with citations (T3), LLM-judge routing + feedback loop + threshold tuning + rate limits (T4). 227 tests pass as of the EC-XGB refresh.

### B9 — EC-XGB (evidence-consistent XGBoost) — ✅ DONE (2026-09-25)
`run_b6_modified.py` (working file name; artifacts under `artifacts/{models,results}/b6/`) implements the m0-m3 variants with grouped tuning, seeds 42/123/456, strict recall-under-5%-FPR operating points, and shift evaluation. Results: in-domain F1 0.9844 to 0.9855 (not significant, McNemar p=0.45); RAGTruth all-task flag rate 99.9% to 61.3%, AUROC 0.475 to 0.582, ECE 0.635 to 0.278; FaithBench flag rate 99.5% to 7.3%, AUROC 0.509 to 0.594, F1 drops (conservative regime, reported as a limitation). The claim splitter is conjunction-aware and atomic (`src/claims/decompose.py`), which removes compound-claim false contradictions, and verdicts are decided support-first so a spurious cross-sentence contradiction cannot override a strong entailing sentence. A relevance gate also drops retrieved sentences that do not cover the claim vocabulary, so off-topic web pages cannot decide verdicts. `fit_ec_xgb_calibrator.py` fits the deployed Platt display calibrator (QA ECE 0.726 to 0.126). The API serves it as `b6-ec-xgb-v1.0`; the web compare card shows EC-XGB beside the standard baselines.

### B7.6 — Evidence-domain display score (superseded by B9 for the deployed score) — ✅ DONE (2026-08-10)
`calibrated_score` no longer saturates at 99% on full-sentence inputs. `src/models/fit_display_calibrator.py` fits the B4 display calibrator (isotonic) on 5,034 natural RAGTruth QA rows (ECE 0.133 on the disjoint 900-row test, vs 0.819 raw); `/verify` further adjusts the score from per-claim verdicts (contradicted up, supported down). Verified live: grounded full-sentence answer 0.31 (low), contradicted-claim answer 0.80 (high). (The deployed score now uses the EC-XGB Platt calibrator; see B9.)

### B8 — Manuscript & delivery — ✅ DONE (2026-08-10/11)
`report/paper.tex` rewritten as a modular document (intro, literature review, methodology, experimental setup, results, system, discussion, conclusion, reproducibility) with only verified Version B numbers, student-register prose (`report/paper_prompt.md`), hyperlinked citations, and vendored figures (pipeline infographic, system architecture, reliability/calibration/transfer diagrams). `report/proposal.tex` and `proposal.pdf` updated to Version B. All five 2026 citations verified against live sources and fixed in `ref.bib` (Luna authors, IJERT authors, IEEE TAI authors, Multimedia authors, SpikeScore ICLR). Later addition: the journal-format manuscript `Journal_Paper/halurisc.tex` (Elsevier CAS single-column, six-section skeleton, 25 references verified 2026-09-17, 14-page clean build) with the length-confound table and the audit paragraph.
_(Superseded 2026-09-20: the course-format sources were retired, its figures and screenshots moved into `Journal_Paper/`, and the journal manuscript is now the single submission artifact. The paths above describe the state at the time.)_

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
2. Click through every DOI/URL at submission time in `Journal_Paper/ref.bib` (30 entries, verified 2026-09-17 and 2026-09-23); the `ieeeTai` DOI must be re-checked on ieeexplore, which blocks scraping.
3. If desired, update the System screenshots (`Journal_Paper/screenshots/*.png`) to the latest chat auto-risk card, then rebuild: `latexmk -pdf -outdir=out halurisc.tex` (from `Journal_Paper/`).
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


---

## 18. Review-driven revision plan (2026-10-20, ACTIVE)

> **Trigger.** A new external pre-submission review of `ML_Paper.pdf` (17 pages) was received on 2026-10-20. Verdict: **major revision before submission**. This section supersedes §17 for the next cycle and folds in the earlier nine-item checklist. It is a plan, not a completion record; tick boxes as evidence lands.
>
> **Scope note.** The reviewer evaluated the manuscript, not the code. Every claim below therefore has to be made true *in the paper*, with the artifact that supports it.

### 18.1 Why this plan is ordered the way it is

The previous plan (neural baseline, claim-level/full-system evaluation, per-baseline cost, full-pipeline latency) covers only about **25–30%** of this review. The review's Priority 1 items are different:

1. The length-balanced evaluation does not prove shortcut robustness.
2. Calibration and threshold protocols are mixed across sections.
3. ECE values are correct but not labeled by model, split, and calibration protocol in one place.
4. HaluEval's length–label correlation limits what the headline F1 demonstrates.
5. The LLM-judge comparison is too narrow for a general superiority claim.
6. Zero-shot, novelty, latency, and cost need tighter definitions.

The critical path is therefore **validity and claim discipline first**, not new baselines. Adding a fine-tuned encoder will not fix any of items 1–4. The neural baseline stays in the plan as Priority 2 and is scheduled for the 16 GB GPU.

**Operating rule for this cycle**

- All new work runs locally. The RTX 3060 6 GB handles everything except the neural fine-tune; the 16 GB GPU (expected after about one week) handles that.
- Colab is a fallback only. Do not block the plan on it.
- No new paid API dependency is added to the deployed path. GPT-5.6 Luna stays optional for the existing judge comparison; local Qwen 3.5 4B covers the second-judge and claim-verifier roles at zero API cost.
- Every new number must come from a committed artifact under `artifacts/results/review/` (new namespace) or an existing frozen artifact. Never overwrite the frozen B2–B6 artifacts.
- Keep the manuscript at 17 pages unless the chosen venue enforces 16. Confirm the limit before final layout.

### 18.2 Assets, requirements, and what is actually new

#### 18.2.1 Already local (no acquisition cost)

| Asset | Location | Used for |
|---|---|---|
| HaluEval QA | `data/raw/`, `data/processed/` | Training, in-domain test, length analysis |
| RAGTruth official | `data/raw/`, `data/processed/` | Shift, multi-source training, target calibration |
| FaithBench | `data/raw/`, `data/processed/` | Summarization shift check |
| EC-XGB and B2 baselines | `artifacts/models/` | All comparisons |
| NLI DeBERTa-v3 cross-encoder | HF cache | Claim features, claim verifier |
| MiniLM embeddings, spaCy | HF cache / `.venv` | Semantic and entity features |
| Per-seed predictions and metrics | `artifacts/results/` | Re-analysis without retraining |
| FastAPI `/verify` pipeline | `src/api/main.py` | Full-system latency/cost |

#### 18.2.2 New compute

| Resource | When | Use |
|---|---|---|
| RTX 3060 6 GB | Now | Re-analysis, calibration tables, XGBoost ablations, Qwen 4-bit judge |
| 16 GB GPU | After ~1 week | DeBERTa-v3-base fine-tune at 512 tokens, 3 seeds |
| Local Qwen 3.5 4B | Now | Second judge, claim verifier, optional judge fallback in `/verify` |
| CPU only | Now | All P1 validity work, writing, LaTeX, tests |

**Qwen 3.5 4B serving.** Use Ollama or vLLM behind an OpenAI-compatible endpoint. Ollama is the simplest on Windows:

```text
ollama list          # confirm the installed Qwen 3.5 4B tag
ollama serve         # OpenAI-compatible base: http://localhost:11434/v1
```

On the 6 GB card use a 4-bit quantized build (about 2.5–3 GB); on 16 GB use fp16 (about 8 GB) or the same quantized build for consistency. Record the exact tag and quantization in the paper.

**Qwen limits.** A 4B model is a *second judge*, not a replacement for GPT-5.6 Luna and not ground truth. Report it separately, with its prompt, decoding, and agreement against the GPT judge and the human audit. Do not use Qwen labels to train EC-XGB or to define the test labels.

#### 18.2.3 API needs (optional, capped)

| API | Needed? | Cost | Rule |
|---|---|---|---|
| GPT-5.6 Luna | Optional, existing judge comparison and one second prompt | $0.105 per 1,000 predictions measured; 200 samples ≈ $0.021 | Cap at $1 for the revision |
| Brave Search | No for the paper evaluation | $0 if using provided contexts; paid only for live UI tests | Do not use live search to produce paper numbers |
| Tavily | No | Same | Legacy fallback only |
| OpenAI embeddings | No | $0 | MiniLM is local |

#### 18.2.4 New dataset decision

**No new dataset is required for Priority 1.** The review asks for an "independently sourced dataset with a different length–label relationship" under the length-shortcut item. The paper already has two external corpora (RAGTruth and FaithBench) whose length–label relation differs from HaluEval; the honest result is that the model transfers poorly. That negative result is the external check.

Plan:

1. First use the existing external corpora plus a length-matched HaluEval subset (WS-B). This is sufficient to soften the claim and answer the reviewer.
2. If the target venue or reviewer insists on a *new* independent set, add **one** free, license-clean QA or summarization set with natural responses. Check the license and label construction before downloading; record the revision hash in `docs/manifest.frozen.json` and the dataset registry. Do not bundle restricted data.
3. If no suitable set is available within the schedule, state it as a limitation. The reviewer says "where feasible"; a documented limitation is acceptable, an invented dataset is not.

Do not acquire a new dataset before WS-A and WS-B are done. They may already close the item.

### 18.3 Workstreams

Status key: `TODO` not started, `WIP` in progress, `DONE` complete. Every workstream has an artifact and an acceptance test.

---

#### WS-A — Calibration and threshold protocol reconciliation

**Review mapping:** critical issues 4 and 5; Section 8; abstract and conclusion.
**Priority:** P1. **Time:** 3–4 working days. **Compute:** CPU. **Depends on:** nothing.

**Why.** The paper reports 0.726 → 0.126 for the EC-XGB display calibrator, 0.8185 → 0.1335/0.1316 for the standard model's target calibrators, and 0.7433 for EC-XGB raw RAGTruth QA ECE. Each value can be correct and still confuse a reader. The reviewer also sees two threshold stories: a strict threshold on HaluEval validation and a 0.313 threshold on the target calibration split. These must become explicit protocols.

**Steps**

- [ ] Locate the source artifact and configuration for every calibration value:
  - 0.726 (EC-XGB raw QA ECE, display calibrator input)
  - 0.126 (EC-XGB display after Platt)
  - 0.7433 (EC-XGB raw QA ECE in Table 6)
  - 0.8185 / 0.1335 / 0.1316 (standard model raw / target Platt / target isotonic)
  - Record checkpoint, seed aggregation, split, calibration fit rows, and test rows.
- [ ] Explain 0.7433 vs 0.726 explicitly. If they come from different seeds, checkpoints, or splits, say so. If they are the same configuration, recompute and fix the prose.
- [ ] Define three protocols in §3.3/§3.4 and use the same names everywhere:
  - **P0 source-only:** calibrator fitted on HaluEval validation, threshold selected on HaluEval validation.
  - **P1 target-calibrated:** calibrator fitted on the 5,034-row RAGTruth QA calibration split, evaluated on the disjoint 900-row test, threshold fixed at 0.5.
  - **P2 target-calibrated + target-thresholded:** threshold selected on the target calibration split at a 5% FPR budget, then evaluated on the 900-row test. Report test FPR, recall, precision, F1 with group-bootstrap 95% CIs.
- [ ] Build **one** combined table: rows = standard XGBoost, EC-XGB; columns = raw, source-calibrated (P0), target-calibrated (P1), target-thresholded (P2); metrics = ECE, Brier, NLL, AUROC, F1, predicted-positive rate. Same 900 rows, same seeds, n in the caption.
- [ ] State that P1 and P2 use labeled target data and are not zero-shot. State that no test label enters fitting or threshold selection.
- [ ] Align Algorithm 2, §3.3, §3.4, §4.4, §5.2, §5.3, abstract, and conclusion.
- [ ] Add one sentence on why 0.126 (EC-XGB) differs from 0.1335 (standard): different upstream model, same protocol, same split.

**Files**

- New: `src/models/review_calibration_protocol.py`
- New artifacts: `artifacts/results/review/calibration_protocol.json`, `calibration_protocol.csv`
- Edit: `Journal_Paper/halurisc.tex` (Table 8 area, §3.3, §3.4, §4.4, §5.2, §5.3, abstract, conclusion)

**Acceptance**

- Every value in the table matches an artifact key exactly (four decimals).
- The table labels model, calibrator, fit split, test split, seed aggregation, and n.
- No sentence uses "zero-shot" for a target-calibrated score.
- No test label is used for fitting or threshold selection.
- The three protocols are distinguishable without reading the code.

---

#### WS-B — Length-shortcut control

**Review mapping:** critical issues 3 and 6; Section 5.8 and Table 14.
**Priority:** P1. **Time:** 2–3 working days. **Compute:** CPU. **Depends on:** WS-D (predictions verified).

**Why.** Equalizing 247 rows per length bin equalizes the *marginal* length distribution, not the label prevalence inside each bin. The current sentence "The model therefore holds when the length mix is neutralized" is stronger than the design supports. The fix is a stronger control and a softer claim.

**Steps**

- [ ] Rebuild the 247 × 6 subset from the saved test predictions and `features_full.parquet`.
- [ ] Report, for every bin: n, positives, negatives, positive rate, mean/median length, and the class-length standardized mean difference (SMD).
- [ ] Evaluate the length-only logistic regression on exactly the same subset. Report F1, AUROC, ECE, and Δ versus EC-XGB with group-bootstrap 95% CIs.
- [ ] Build a **matched subset**: within each bin, match each positive to a negative on exact answer length (caliper 0 words, widen to 1 word if needed), equal counts per bin. Report n, prevalence, SMD, and the same metrics for length-only, standard XGBoost, and EC-XGB.
- [ ] If a bin has fewer than 30 rows after matching, pool adjacent bins or widen the caliper; report the decision.
- [ ] Rewrite §5.8 and Table 14:
  - Call the current test a **length-bin reweighting control**, not proof of shortcut removal.
  - State what it shows (performance is stable when the marginal length mix is equalized).
  - State what it does not show (within-bin label–length association remains).
  - Report the matched-subset result as the stronger control.
- [ ] Optional external check: correlate absolute error or score with answer length on RAGTruth QA and FaithBench. Report whether the shortcut persists out of domain.

**Files**

- New: `src/models/review_length_control.py`
- New artifacts: `artifacts/results/review/length_control.json`, `length_control_bins.csv`, `length_control_matched.csv`
- Edit: §5.8, Table 14 caption, conclusion limitation sentence

**Acceptance**

- Per-bin positives and negatives are in the artifact and the table caption or supplement.
- Length-only is evaluated on the exact same rows as EC-XGB.
- The matched subset's class-length balance is reported (SMD near zero or the residual stated).
- The word "neutralized" and the claim of proven robustness are removed.
- The conclusion keeps the negative result and the length caveat.

---

#### WS-C — Zero-shot and claim language

**Review mapping:** high issue 10; abstract, §3.3, §4.2, §5.4, §5.9, conclusion.
**Priority:** P1. **Time:** 1 working day. **Compute:** CPU. **Depends on:** WS-A wording.

**Why.** The paper says the held-out QA result is zero-shot, but the deployed display score is fitted on 5,034 RAGTruth QA calibration rows. Classification and calibration are different exposure conditions.

**Steps**

- [ ] Add one definition paragraph: "zero-shot" means no target examples, groups, tasks, or sources were used for training, calibration, or threshold selection. Define each axis separately.
- [ ] Rewrite every use:
  - Standard XGBoost external results: zero-shot.
  - EC-XGB combined RAGTruth: seen-task adaptation (summarization and data-to-text were in training).
  - EC-XGB held-out QA classifier: zero-shot at the classifier level.
  - EC-XGB QA display score: target-calibrated, **not** zero-shot.
  - FaithBench: state whether the source or target calibrator is used.
- [ ] Replace "unseen" with a precise axis (examples, groups, tasks, or sources) wherever it appears.
- [ ] Update the abstract and conclusion to match.

**Acceptance**

- No sentence calls a target-calibrated score zero-shot.
- Each result names its exposure condition.
- The abstract separates combined RAGTruth, held-out QA, and FaithBench.

---

#### WS-D — Number and artifact audit

**Review mapping:** Priority 1 "reproduce key numbers"; Section 11.
**Priority:** P1. **Time:** 1–2 working days. **Compute:** CPU. **Depends on:** nothing.

**Steps**

- [ ] Run `python src/models/verify_artifacts.py` and save the output to `artifacts/results/review/verify_artifacts.log`.
- [ ] Build `artifacts/results/review/number_audit.csv` with one row per paper number: paper location, value, artifact file, JSON/CSV key, seed aggregation, split, status (`match`, `explain`, `fix`).
- [ ] Check all tables 4–14 and every headline number: F1, AUROC, PR-AUC, MCC, ECE, Brier, NLL, flag rate, latency, cost, bootstrap intervals, McNemar/Wilcoxon p-values.
- [ ] Reconcile the calibration values from WS-A in the same sheet.
- [ ] Verify split hashes, seeds, requirements pins, and dataset license notes.
- [ ] Do not regenerate or overwrite frozen artifacts. If a value is wrong, fix the prose or add a new review artifact with a note.

**Acceptance**

- Zero untraced numbers.
- Every mismatch is either fixed or documented with the correct configuration.
- The audit sheet is committed.

---

#### WS-E — LLM-judge scope and second judge

**Review mapping:** high issue 7; Table 12, §5.7.
**Priority:** P2. **Time:** 1–2 working days. **Compute:** CPU + optional GPT/Qwen. **Depends on:** WS-D.

**Steps**

- [ ] Document the 200-sample construction: source pool (HaluEval test), sampling seed, class balance, positive prevalence, and whether these rows overlap the headline test set.
- [ ] Add 95% bootstrap CIs for accuracy, precision, recall, and F1 for both the judge and EC-XGB.
- [ ] Evaluate the length-only baseline on the same 200 rows.
- [ ] Record the judge prompt, model identifier (`gpt-5.6-luna`), temperature (0), max tokens, and scoring rule in the paper or supplement.
- [ ] Add a **local Qwen 3.5 4B** second judge on the same 200 rows:
  - Serve via Ollama at `http://localhost:11434/v1`.
  - Add `HALU_JUDGE_BASE_URL` and `HALU_JUDGE_MODEL` plumbing to `src/claims/judge.py` and `src/models/eval_llm_judge.py`.
  - Use the same prompt; record quantization and decoding.
  - Report F1, agreement with GPT, and agreement with the labels; human spot-check 20–40 disagreements.
- [ ] Narrow the claim to "on this 200-sample HaluEval subset with this prompt" rather than general superiority over LLM judges.
- [ ] Optional: one second GPT prompt or a 500-sample replication, capped at $1.

**Files**

- Edit: `src/claims/judge.py` (base URL), `src/models/eval_llm_judge.py` (`--base-url`, `--model`)
- New artifacts: `artifacts/results/review/llm_judge_qwen.json`, `llm_judge_ci.json`
- Edit: Table 12 caption, §5.7, conclusion sentence

**Acceptance**

- Sampling, prevalence, prompt, model, decoding, and CIs are documented.
- Qwen results are labeled as a second judge, not merged with GPT.
- The claim is scoped to the sample and prompt.
- Total API spend for this workstream is under $1.

---

#### WS-F — Novelty, monotone ablation, and closest baselines

**Review mapping:** high issue 9; Introduction and §3.3.
**Priority:** P2. **Time:** 2–3 working days. **Compute:** CPU (XGBoost retrain is fast). **Depends on:** WS-D.

**Steps**

- [x] Rewrite the contribution as a **method + evaluation protocol + reproducible artifact**, not a new learning algorithm.
- [x] Add a separate **monotone-constraint ablation** (`review_monotone_ablation.py`, 3 seeds): F1 unchanged, ECE -0.0009, RAGTruth AUROC +0.0043; §5.2 sentence.
- [x] Add one explicit sentence for the claim aggregates: m1 versus m2 adds the eight claim-level aggregates beyond the six sentence-level NLI probabilities (+0.0005 F1, within undistinguished seed-level variation).
- [x] Cite the existing source-indicator ablation (0.9849 versus 0.9855; shifted flag rate within one point).
- [x] Add a short closest-baseline comparison paragraph (§3.3):
  - Sundaragiri et al. (retrieval + NLI + XGBoost)
  - Valentin et al. (calibrated multi-scoring)
  - SpikeScore (cross-domain)
  - State protocol differences, not only "classification metrics only."
- [x] Do not claim "first" or "novel" without a verified comparison.

**Files**

- New: `src/models/review_monotone_ablation.py`
- New artifacts: `artifacts/results/review/monotone_ablation.json`
- Edit: Introduction contribution paragraph, §3.3, §5.5, Table 5 or appendix

**Acceptance**

- Monotone constraints have a standalone row.
- Claim aggregates have an explicit beyond-NLI statement.
- Closest baselines are compared on protocol and metrics.
- No unsupported novelty claim remains.

---

#### WS-G — Neural baseline (16 GB GPU)

**Review mapping:** high issue 9 and Priority 2 "strengthen the closest-baseline comparison"; old nine-item neural baseline.
**Priority:** P2. **Time:** 2–3 working days. **Compute:** 16 GB GPU (1–2 h training) or 6 GB fallback (3–5 h). **Depends on:** 16 GB GPU arrival; WS-D splits.

**Model choice**

- **16 GB:** `microsoft/deberta-v3-base`, max length 512, batch 16, fp16/bf16, 3 epochs, lr 2e-5, early stopping on validation.
- **6 GB fallback:** `roberta-base` (or DistilRoBERTa), max length 256, batch 8 with gradient accumulation 2, fp16.
- Install `accelerate` (latest compatible with torch 2.11 and transformers 5.14.1), pin exact per AGENTS.md §5.1. Do not bump xgboost.

**Steps**

- [x] Reuse the existing grouped splits and labels. Do not re-split.
- [x] Input format: question + context + answer, truncated to max length; truncation rate 14/3{,}000 test rows (`neural_baseline_truncation.json`).
- [x] Train seeds 42/123/456; save per-seed metrics, validation/test predictions (`neural_baseline_predictions.parquet`). Weight checkpoints are not kept; retraining from the pinned script reproduces them.
- [x] Fit Platt calibration on the validation split only; evaluate in-domain and on RAGTruth QA, RAGTruth all, and FaithBench.
- [ ] Report F1, AUROC, PR-AUC, MCC, ECE, Brier, NLL, parameter count, latency, and cost per 1,000.
- [ ] Optional multi-source variant: train on HaluEval + RAGTruth non-QA rows, matching EC-XGB's exposure, to separate model family from training data.
- [x] Add one row to Table 5 (or a new table) and one paragraph. If the neural model wins in-domain, frame EC-XGB on calibration, cost, and interpretability; if it does not, report the result without spin.

**Files**

- New: `src/models/train_neural_baseline.py`
- New artifacts: `artifacts/results/review/neural_baseline/` (metrics, predictions, calibration)
- Edit: Table 5, §5.1, §5.5, conclusion

**Acceptance**

- Same splits, same labels, three seeds.
- Calibration fitted on validation only.
- Per-example predictions saved.
- Latency and cost measured on the same machine.
- The comparison is honest about the 6 GB versus 16 GB hardware difference.

---

#### WS-H — Latency and cost accounting

**Review mapping:** high issue 11; Table 11, §5.7, abstract, §3.5.
**Priority:** P2/P3. **Time:** 1–2 working days. **Compute:** CPU + optional judge. **Depends on:** WS-D.

**Steps**

- [x] Build a per-baseline table: input requirements, model size, training hardware (if any), inference hardware, p50/p95 latency, cost per 1,000 predictions. (Table `tab:cost` with model-only p50 + cost/1K; training hardware in §4.3 for RoBERTa; inference hardware in table intro.)
- [x] Measure the full `/verify` pipeline end to end (`review_pipeline_latency.py`): predict p50 42.8 ms, analyze 57.1 ms, verify 143.6 ms, verify+judge 502.5 ms; §5.7 sentence.
- [x] Report classifier-only latency (61.8 ms) separately from the HTTP path, with loading/retrieval/judge exclusions stated in §5.7.
- [x] State cost assumptions in §5.7: amortized local hardware + electricity (order-of-magnitude USD/1K); GPT token pricing $0.20/$1.20 per MTok in/out.
- [x] Update §5.7 (Table `tab:cost`, exclusions, assumptions) so no sentence conflates the two latencies.

**Acceptance**

- Classifier-only and end-to-end numbers are separate.
- p50 and p95 are reported for the end-to-end path.
- Assumptions and units are explicit.
- No "actionable" claim without a decision policy.

---

#### WS-I — SHAP stability versus faithfulness

**Review mapping:** medium issue 12; §3.4, §5.6, Figure 6, Table 10.
**Priority:** P3. **Time:** 1 working day. **Compute:** CPU. **Depends on:** WS-B (length caveat).

**Steps**

- [ ] Define stability (rank correlation, top-k Jaccard across bootstrap/perturbation) and faithfulness (does the explanation reflect the model's actual evidence use) as separate properties.
- [ ] Document the top-k neutralization procedure: baseline value, k, sample size, model output used, and the metric.
- [ ] State the small-sample limits for dates (n = 13) and clause shuffles (n = 14).
- [ ] Add one sentence: a stable top feature can be a shortcut such as answer length; stability does not prove semantic faithfulness.
- [ ] Avoid saying the explanations are "correct" or "faithful" without a human or causal check.

**Acceptance**

- The paper distinguishes stability from faithfulness.
- Procedure and sample sizes are explicit.
- The length shortcut is linked to the explanation caveat.

---

#### WS-J — Tables, figures, abstract, conclusion, and page budget

**Review mapping:** Section 13, Section 14, Priority 4.
**Priority:** P3/P4. **Time:** 2 working days. **Compute:** CPU. **Depends on:** WS-A to WS-I.

**Steps**

- [ ] Add n and positive prevalence to key evaluation tables.
- [ ] Add group-bootstrap CIs for central cross-domain results (held-out QA AUROC, RAGTruth AUROC, FaithBench F1).
- [ ] Label every calibration plot by model and split (raw, source, target).
- [ ] Adopt the reviewer's conservative framing in the abstract and conclusion:
  - F1 0.9855 is HaluEval QA benchmark performance.
  - Combined RAGTruth and held-out QA are separate.
  - ECE 0.726 → 0.126 belongs to the EC-XGB display calibrator; target calibration needs labels.
  - The length control is a control, not proof.
  - No general superiority over LLM judges.
  - "Risk estimation," not truth.
- [ ] Preserve the negative results: held-out QA F1 ≈ 0.30 and AUROC ≈ 0.58 (weak discrimination).
- [ ] Page budget: current 17 pages. Expected additions: combined calibration table (~0.4 page), length control table (~0.4), monotone ablation (~0.3), neural baseline row (~0.2), cost table (~0.4), text (~1.0), total +2.7 pages. Compress by merging tables, moving details to a supplement, tightening §2–§3 prose, and keeping figures at their current count. Confirm whether the venue allows 17 pages or enforces 16 before trimming.
- [ ] Final language pass: no pronouns, semicolons, em-dashes, `etc.`, or banned words; keep median sentence length near 17 words.
- [ ] Rebuild: `latexmk -pdf -outdir=out halurisc.tex` from `Journal_Paper/`. Check 0 overfull, 0 underfull, 0 undefined citations.

**Acceptance**

- 17 pages (or the venue limit), zero warnings, all references cited.
- All tables have n and prevalence where relevant.
- The abstract and conclusion match the evidence.
- Figures remain readable at print size.

---

#### WS-K — Optional claim-level and full-system evaluation (deferred)

**Review mapping:** old nine-item checklist; **not** required by the new review.
**Priority:** P3, deferred. **Time:** 3–5 working days. **Compute:** CPU + Qwen. **Depends on:** WS-J.

Do this only if the target venue asks, or if there is time after the P1/P2 work.

- [ ] `python src/models/eval_claims.py --build --n 40`; label 40–100 claims by hand (1–2 h).
- [ ] Evaluate NLI verdicts against the human labels; add Qwen as a second claim verifier.
- [ ] Run `/verify` on a held-out sample with local Qwen as the judge fallback; report end-to-end results and latency.
- [ ] Keep claim-level and full-system results separate from the classifier results.

---

### 18.4 Traceability matrix

| Review item | Workstream | Evidence to add | Status |
|---|---|---|---|
| §3 length-balanced logic | WS-B | Per-bin counts, matched subset, length-only on same rows, CIs, softened claim | TODO |
| §4 calibration/threshold protocol | WS-A | Three protocols, one combined table, aligned sections | TODO |
| §5 model-specific calibration labels | WS-A | Combined standard vs EC-XGB table; explain 0.7433 vs 0.726 | TODO |
| §6 HaluEval length confound | WS-B, WS-F | Length control, length-only margin, claim discipline | TODO |
| §7 LLM-judge narrowness | WS-E | Sampling, prevalence, CIs, prompt, Qwen second judge, scoped claim | TODO |
| §8 operating-point interpretation | WS-A | Validation target vs test FPR, CIs, source vs target threshold | TODO |
| §9 novelty | WS-F, WS-G | Method/protocol framing, monotone ablation, closest baselines, neural row | TODO |
| §10 zero-shot terminology | WS-C | Definitions and rewritten claims | TODO |
| §11 latency/cost | WS-H | Per-baseline and full-pipeline table, explicit assumptions | TODO |
| §12 SHAP reliability | WS-I | Stability vs faithfulness, procedure, sample sizes | TODO |
| §13 tables/figures/equations | WS-J | n/prevalence, CIs, labels, terminology | TODO |
| §14 abstract/conclusion | WS-J | Conservative framing, negative results preserved | TODO |
| §15 P1 | WS-A to WS-D | All critical fixes with artifacts | TODO |
| §15 P2 | WS-E to WS-G | Judge, ablations, neural baseline | TODO |
| §15 P3 | WS-H to WS-J | Cost, SHAP, tables, reproducibility | TODO |
| §15 P4 | WS-J | Final language and format pass | TODO |
| Old: per-baseline cost | WS-H | Table | TODO |
| Old: full-pipeline latency | WS-H | p50/p95 by stage | TODO |
| Old: neural baseline | WS-G | Table row + paragraph | TODO |
| Old: claim-level/full-system | WS-K | Optional, deferred | DEFERRED |

### 18.5 Timeline and milestones

Assumes 4–6 focused hours per working day; the GPU wait overlaps with P1.

| Window | Work | Output |
|---|---|---|
| Days 1–2 | WS-D audit, WS-A calibration source tracing | `number_audit.csv`, calibration configs |
| Days 3–5 | WS-A table and protocol text, WS-B per-bin counts and matched subset | Combined calibration table, length control artifacts |
| Days 6–7 | WS-C zero-shot rewrite; WS-B prose; finish P1 | P1 package ready for internal read |
| Day 7–8 (GPU arrives) | WS-G training and evaluation; WS-F monotone ablation | Neural baseline + monotone results |
| Days 9–10 | WS-E judge details and Qwen second judge; WS-H cost/latency | Judge and cost tables |
| Days 11–12 | WS-I SHAP; WS-J tables, abstract, conclusion | Draft with all new evidence |
| Days 13–14 | Page fitting, full read, rebuild, tests, checklist update | Submission draft |
| Day 15 | Final DOI/URL check and push | Submitted PDF + tagged commit |

**Milestones**

- **M1 (day 5):** P1 validity package done. Calibration table, length control, zero-shot wording, number audit.
- **M2 (day 8):** Neural baseline and monotone ablation done.
- **M3 (day 10):** Judge, cost, and latency evidence done.
- **M4 (day 12):** Full draft with all tables.
- **M5 (day 14):** 17-page clean build, tests pass, checklist updated.
- **M6:** submit.

### 18.6 Cost model (local-first)

| Item | Assumption | Cost |
|---|---|---|
| Electricity | 10–20 GPU-hours, ~300 W, local tariff | $0.35–1.20 |
| GPT-5.6 Luna judge | 200 samples measured at $0.021; cap 1,000 samples | $0.02–0.11 |
| Second GPT prompt (optional) | 200 samples | $0.02 |
| Qwen 3.5 4B | Local | $0 |
| Brave/Tavily | Not used for paper numbers; provided contexts only | $0 |
| Live search smoke test (optional) | 100–200 queries, free tier if available | $0–5 |
| Human claim labels | 40–100 claims, 1–2 h | $0 cash |
| **Total** | | **$0–6.50, typical < $2** |

No cloud GPU rental. If the 16 GB card is delayed, the 6 GB fallback (RoBERTa-base at 256 tokens) keeps the neural baseline feasible; the training takes 3–5 hours instead of 1–2.

### 18.7 What remains after this plan

Expected coverage: about **80–85%** of the new checklist, including all five Priority 1 items and most Priority 2 items.

Remaining and how to handle them:

1. **A truly independent dataset with a different length–label relation.** Use existing RAGTruth and FaithBench plus the matched subset first. If still required, add one free, license-clean set with a documented hash; otherwise state the limitation.
2. **A second judge or prompt.** Qwen 3.5 4B covers the second judge. A second GPT prompt is optional and capped at $1.
3. **Claim-level and full-system evaluation.** Not requested by this reviewer. Keep as future work unless the venue asks (WS-K).

If only the old plan is executed (neural baseline + cost/latency + claim-level), coverage stays near 25–30% and the "major revision" verdict stands.

### 18.8 Page budget

| Addition | Estimate |
|---|---|
| Combined calibration table | +0.4 page |
| Length control table and matched-subset row | +0.4 |
| Monotone-constraint ablation row | +0.3 |
| Neural baseline row and paragraph | +0.2 |
| Per-baseline cost table | +0.4 |
| Rewritten protocol/zero-shot/SHAP text | +1.0 |
| **Gross addition** | **+2.7 pages** |
| Trims: merge Tables 6 and 8, move per-bin counts to supplement, tighten §2–§3 | −2.0 to −3.0 |
| **Target** | **16–17 pages** |

Confirm the venue limit before deciding whether to trim to 16. Do not cut the negative results or the calibration protocol to save space.

### 18.9 Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Calibration values cannot be reconciled | Medium | Trace every value to its artifact first; if a configuration is missing, re-run only that fit and save a new review artifact. Do not overwrite frozen files. |
| Matched length subset too small | Medium | Widen the caliper, pool adjacent bins, report n and CIs; keep the reweighting control as the secondary result. |
| 16 GB GPU delayed | Medium | Do P1 first; train RoBERTa-base on the 6 GB card as fallback. |
| Qwen 4B judge quality is low | Medium | Use it as a second judge only; report agreement, not a headline F1; human spot-check disagreements. |
| Extra tables exceed 17 pages | High | Merge tables, move details to supplement, tighten prose; confirm venue limit early. |
| API cost overrun | Low | Cap GPT spend at $1; use Qwen for the second judge. |
| Human claim labels take longer | Medium | Limit to 40 claims; label only the disagreement stratum; report the sampling. |
| Reviewer sees the old PDF | Medium | Rebuild and send the new PDF with a short change log; update `Review_Checklist.md` with evidence links. |

### 18.10 Definition of done

- [ ] All five Priority 1 items closed with a table, paragraph, or artifact.
- [ ] Every number in the manuscript traces to `number_audit.csv`.
- [ ] Calibration table labels model, calibrator, split, seeds, and n.
- [ ] Length claim softened; matched subset and per-bin counts reported.
- [ ] Zero-shot terminology is consistent across abstract, method, results, and conclusion.
- [ ] Neural baseline and monotone ablation are in the paper or supplement.
- [ ] Per-baseline cost and full-pipeline latency are separated and explicit.
- [ ] LLM-judge claim is scoped; Qwen second judge reported separately.
- [ ] Abstract and conclusion use the conservative framing.
- [ ] 17 pages (or venue limit), zero LaTeX warnings, 38 references cited.
- [ ] `pytest` passes, web build/lint pass if web files change.
- [ ] `Review_Checklist.md` updated with the new evidence and open items.
- [ ] PDF rebuilt, committed, and pushed.

### 18.11 Commands and artifact paths

Existing commands:

```powershell
& .venv\Scripts\python.exe src\models\verify_artifacts.py
& .venv\Scripts\python.exe src\models\analyze_length_shortcut.py
& .venv\Scripts\python.exe src\models\review_extras.py
& .venv\Scripts\python.exe src\models\eval_llm_judge.py
& .venv\Scripts\python.exe src\models\eval_efficiency.py
& .venv\Scripts\python.exe src\models\eval_claims.py --build --n 40
& .venv\Scripts\python.exe -m pytest tests -v
```

New scripts to create:

```text
src/models/review_calibration_protocol.py
src/models/review_length_control.py
src/models/review_monotone_ablation.py
src/models/train_neural_baseline.py
src/models/review_cost_table.py
```

New artifact namespace:

```text
artifacts/results/review/
  calibration_protocol.json
  calibration_protocol.csv
  length_control.json
  length_control_bins.csv
  length_control_matched.csv
  monotone_ablation.json
  neural_baseline/
  llm_judge_qwen.json
  llm_judge_ci.json
  number_audit.csv
  verify_artifacts.log
```

Manuscript build:

```powershell
Set-Location Journal_Paper
latexmk -pdf -interaction=nonstopmode -outdir=out halurisc.tex
```

Qwen environment (optional, local only; never commit keys):

```text
HALU_JUDGE_BASE_URL=http://localhost:11434/v1
HALU_JUDGE_MODEL=<installed-qwen3.5-4b-tag>
HALU_JUDGE_API_KEY=ollama
```

### 18.12 What not to do

- Do not start with the neural baseline. It is Priority 2 and does not fix the critical validity issues.
- Do not call the length-bin reweighting a neutralization or proof of shortcut robustness.
- Do not use test labels for calibration or threshold selection.
- Do not call a target-calibrated QA score zero-shot.
- Do not claim general superiority over LLM judges from 200 samples and one prompt.
- Do not add a second dataset just to have one; use the existing external corpora and matched controls first.
- Do not overwrite frozen artifacts or bump xgboost (stays 3.3.0).
- Do not add references without live verification.
- Do not exceed 17 pages without checking the venue limit. (2026-10-21: author approved 18 pages for the second review response.)

### 18.13 Second external review (2026-10-21) — closed

> A follow-up review raised four points. Each was checked against the manuscript before acting.

1. **Calibration numbers inconsistent.** Valid but narrow. `0.726 → 0.126` is the seed-42 deployed fit, Table 9 is three-seed means (`0.7433 → 0.1263`). Fixed in §5.3 with one clause naming both sources.
2. **Length shortcut.** Partly valid. The 158-row control was nested inside the 1,482-row reweighted subset. New `review_length_matched_repeat.py` matches on the full test set (163 pairs, 326 rows), repeats the matching 200 times, and adds a length-conditional AUROC over 12 exact-length strata. Numbers in `length_matched_repeat.json`; §5.8 rewritten.
3. **Weak generalization.** Already stated in the abstract, §5.2, and §5.3. Added one explicit sentence in the Discussion naming EC-XGB a domain-specific estimator rather than a general-purpose detector.
4. **Zero-shot versus target-calibrated.** Already separated in §3.3, §3.4, §4.4, §5.3, and the Table 5 caption. Added Table 4 (`tab:protocols`) with P0/P1/P2 against calibrator fit, threshold fit, and zero-shot status.

**Outcome:** 17 pages after shortening the conclusion and trimming duplicated protocol text, zero warnings, number audit 200/200, 231 tests pass. Citation rule from the author: one reference per claim, two at most, never three. The only triple citation was split into two sentences.
