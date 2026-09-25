# HaluRISC — Calibrated & Explainable Hallucination Risk Analyzer

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-green.svg)](https://fastapi.tiangolo.com/)
[![assistant-ui](https://img.shields.io/badge/assistant--ui-0.15-violet.svg)](https://www.assistant-ui.com/)
[![Tailwind v4](https://img.shields.io/badge/Tailwind-v4-38bdf8.svg)](https://tailwindcss.com/)

**HaluRISC** (Hallucination Risk Scoring and Calibration) is a lightweight, black-box machine learning framework for predicting hallucination risk in Large Language Model (LLM) outputs. It combines evidence-aware text features, calibrated tree ensembles (XGBoost), SHAP feature attributions, and a GPT 5.6 Luna-powered conversational AI chat interface built with **`assistant-ui`**.

---

## 🌟 Key Features

- 💬 **Conversational AI Risk Analyst (`assistant-ui`)**: Interactive streaming chat powered by GPT 5.6 Luna that explains why an answer is risky in natural language.
- 🎨 **Generative UI Widgets**: Animated SVG semicircular risk gauges and SHAP feature contribution charts rendered directly inside chat messages.
- 📊 **Multi-Mode Web App**:
  - **💬 Chat Mode**: Conversational risk interrogation (`/chat`)
  - **📊 Analyze Mode**: Form-based evidence inspector (`/analyze`)
  - **📈 Dashboard**: Empirical benchmarks and cost comparisons (`/dashboard`)
  - **ℹ️ About**: Pipeline architecture and method overview (`/about`)
- ⚡ **Lightweight & Fast**: 61.8 ms per analysis at the median over 200 test samples (XGBoost inference 1.3 ms, SHAP 1.9 ms).
- 💰 **Cheaper and Faster than LLM Judges**: measured $0.105/1K predictions and 1,310 ms for GPT 5.6 Luna as judge, versus about $0.001/1K and 61.8 ms for the local model — roughly 100x cheaper and 20x faster.
- 🔬 **Statistically Rigorous**: 20,000 samples (HaluEval QA), leakage-free grouped 70/15/15 splits, 3-seed protocol (42/123/456), grouped 5-fold CV tuning, McNemar tests and bootstrap CIs. Raw probabilities are already calibrated on the source domain (ECE 0.0045); Platt (0.0091) and isotonic (0.0070) do not improve them there.
- 🌍 **Cross-Domain and Explanation-Aware**: zero-shot transfer measured on RAGTruth and FaithBench, target-domain recalibration (ECE 0.819 → 0.134 on RAGTruth QA), SHAP reliability metrics, a structured expert audit of 26 error-enriched cases, and an answer-length confound analysis.
- 📄 **Journal Manuscript**: an Elsevier CAS single-column manuscript (`Journal_Paper/halurisc.tex`), built from the frozen artifacts.

---

## 📊 Benchmark Results (HaluEval QA test set, N = 3,000)

Leakage-free grouped split; mean over seeds 42/123/456, from
`artifacts/results/b2/b2_model_comparison.csv`:

| Model Architecture          | Precision  | Recall     | F1-Score   | AUROC      | PR-AUC     | MCC        |
| --------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **Majority (all 0)**        | 0.0000     | 0.0000     | 0.0000     | n/a        | n/a        | 0.0000     |
| **Heuristic (1 - overlap)** | 0.9473     | 0.9233     | 0.9352     | 0.9132     | 0.8196     | 0.8723     |
| **TF-IDF (Q+C+A)**          | 0.6397     | 0.5647     | 0.5999     | 0.6754     | 0.6785     | 0.2484     |
| **TF-IDF (answer only)**    | 0.9550     | 0.8920     | 0.9224     | 0.9696     | 0.9665     | 0.8519     |
| **TF-IDF (context only)**   | 0.5000     | 1.0000     | 0.6667     | n/a        | n/a        | 0.0000     |
| **NLI-only**                | 0.6479     | 0.6967     | 0.6714     | 0.7177     | 0.7531     | 0.3189     |
| **Logistic Regression**     | 0.9768     | 0.9553     | 0.9660     | 0.9932     | 0.9901     | 0.9329     |
| **Random Forest**           | 0.9897     | 0.9789     | 0.9842     | 0.9980     | 0.9983     | 0.9687     |
| **XGBoost (ours)**          | **0.9932** | **0.9762** | **0.9846** | **0.9979** | **0.9983** | **0.9697** |

**Artifact controls:** answer-only TF-IDF already reaches F1 0.9224 — a large
part of the HaluEval signal is answer-style surface text. Context-only has zero
signal (paired answers share context, F1 0.6667 = always-positive), and
NLI-only is weak (0.6714). The full evidence-aware XGBoost model adds a
significant margin over the controls (McNemar p = 1.3e-06 vs answer-only,
p = 0.044 vs random forest; bootstrap 95% CI F1 [0.9812, 0.9895], AUROC
[0.9964, 0.9989]).

**Deployed model (EC-XGB, B9):** the served model adds monotone evidence
constraints, eight claim-level NLI aggregates over atomic clauses, and RAGTruth
non-QA multi-source training (35 features). It preserves in-domain F1 (0.9855),
cuts the RAGTruth all-task flag rate from 99.9% to 61.3% (AUROC 0.475 -> 0.582,
ECE 0.635 -> 0.278), and its Platt display calibrator drops the RAGTruth QA ECE
from 0.726 to 0.126. Verdicts are support-first and gated by evidence
relevance, so off-topic retrieved pages cannot decide a claim. Contract:
`b6-ec-xgb-v1.0`; artifacts: `artifacts/results/b6/`.

**Calibration on the source domain:** raw XGBoost probabilities are already
well calibrated (ECE 0.0045), and Platt (0.0091) and isotonic (0.0070) do not
improve them — an honest negative result. Calibration pays off only after the
domain shift (see the RAGTruth recalibration below).

**Leakage-removal impact:** the historical row-level split inflated F1 to
0.9886; the grouped split corrects it to 0.9846 (Δ −0.004 in F1, AUROC 0.9979).
The earlier course-run reference with row-level CV tuning was F1 0.9842.
Details: `artifacts/results/b2/b2_leakage_comparison.json`.

### LLM-as-Judge comparison (200 test samples, measured)

| Model              | Accuracy   | Precision  | Recall     | F1         | Latency p50 | Cost / 1K |
| ------------------ | ---------- | ---------- | ---------- | ---------- | ----------- | --------- |
| GPT 5.6 Luna judge | 0.8600     | 0.9740     | 0.7400     | 0.8410     | 1,310 ms    | $0.105    |
| **XGBoost (ours)** | **0.9850** | **0.9800** | **0.9900** | **0.9850** | 62 ms       | ~$0.001   |

Agreement between judge and XGBoost: 0.845 (McNemar p = 1.6e-05).

### External zero-shot validation (no training or adaptation)

| Corpus         | Rows     | F1    | AUROC | ECE   |
| -------------- | -------- | ----- | ----- | ----- |
| RAGTruth QA    | 900      | 0.302 | 0.540 | 0.819 |
| RAGTruth all   | 17,790   | 0.603 | 0.497 | 0.560 |
| FaithBench     | 750      | 0.813 | 0.531 | 0.301 |

Recall is 1.0 on every corpus: the HaluEval-trained model flags almost
everything as risky on natural responses. The length-binned analysis shows why:
in HaluEval the hallucinated share rises from 2.1% for one-word answers to
99.2% above 17 words, and the model reproduced that confound. This is the
central negative result, and it motivates the claim-level evidence verification
in the deployed system. Target-domain recalibration on 5,034 RAGTruth rows cuts
the ECE on the disjoint 900-row test from 0.819 to 0.134.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Next.js Frontend                       │
│      App Router (Port 3000) + assistant-ui + Tailwind       │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
      POST /api/chat                 GET /api/ml/*
               │                      (Proxy rewrites)
               ▼                              │
┌──────────────────────────────┐              ▼
│     GPT 5.6 Luna (OpenAI)    │    ┌─────────────────────────┐
│   Streaming AI SDK Provider  │    │     FastAPI Backend     │
│   Executes analyze tool ─────┼───►│       (Port 8000)       │
└──────────────────────────────┘    │  XGBoost + SHAP + NLI   │
                                    └─────────────────────────┘
```

---

## 🚀 Quick Start Guide

### Prerequisites

- **Python 3.12** (in `.venv`, packages pinned in `requirements.txt`)
- **Node.js 20+ & pnpm** (web app: Next.js 16 + Tailwind v4 + assistant-ui)
- **OpenAI API Key** (for conversational chat & LLM judge baseline)
- Optional NVIDIA GPU (CUDA 12.8, e.g. RTX 3060 6GB) — used for fast feature extraction/training **and** fast API inference (models load in fp16 on CUDA; set `HALU_API_DEVICE=cuda`, auto-falls back to CPU if torch lacks CUDA)

### Reproduced environment (for the paper's reproducibility statement)

- Windows 11, Python 3.12, NVIDIA RTX 3060 6GB (CUDA 12.8, torch 2.11.0+cu128), 32 GB RAM
- scikit-learn 1.9.0, xgboost 3.3.0, shap 0.52.0, spacy 3.8.14 (en-core-web-sm 3.8.0), sentence-transformers 5.6.1

---

### 1. Backend Setup & Data Pipeline

```powershell
# 1. Clone the repository
git clone https://github.com/Atik203/HaluRISC.git
cd HaluRISC

# 2. Activate virtual environment (or create one)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install Python dependencies (pinned, includes spaCy en_core_web_sm model)
pip install -r requirements.txt

# 4. Download HaluEval QA dataset (10,000 JSONL records)
python src/data/download.py

# 5. Process binary dataset (20,000 rows) & generate 70/15/15 splits
python src/data/prepare.py

# 6. Extract full feature matrix (26 features, 7 groups — downloads NLI/SBERT/spaCy models)
python src/features/extract_features.py

# 7. Full experiment protocol: tuning, 3 seeds, calibration, stats, ablations
python src/models/train_pipeline.py

# 8. SHAP explanations + figures (PNG + PDF), saves shap_explainer.joblib
python src/explain/shap_analysis.py

# 9. RAGTruth zero-shot external validation
python src/data/download_ragtruth.py
python src/models/eval_ragtruth.py

# 10. Error analysis (10 FP + 10 FN) and efficiency/latency analysis
python src/models/error_analysis.py
python src/models/eval_efficiency.py

# 11. LLM-as-judge comparison (200 samples, ~$0.02 — optional, needs OPENAI_API_KEY in .env)
python src/models/eval_llm_judge.py

# 12. Start FastAPI inference backend server (Port 8000)
#   Root .env options: HALU_API_DEVICE=cuda|cpu (default cpu; cuda auto-falls back
#   to cpu if torch has no CUDA), HALU_API_PRELOAD=0 (skip heavy-model preload at
#   startup). Run WITHOUT --reload (uvicorn's file watcher restarts on file changes).
& .venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

---

### Unified dataset layer

Canonical, lossless schema for HaluEval + RAGTruth (official) + FaithBench.
The HaluEval preprocessing in `prepare.py` stays as is; the unified layer adds
adapter modules. FaithBench (CC BY-NC-SA) is never bundled — raw files stay
under gitignored `data/raw/`; only hashes, citations, and license notes ship.

```powershell
# 13. Download official RAGTruth (response.jsonl + source_info.jsonl, lossless spans/tasks)
& .venv\Scripts\python.exe src\data\download_ragtruth.py

# 14. Download FaithBench annotation batches (batch_1..16, no batch 13)
& .venv\Scripts\python.exe src\data\download_faithbench.py

# 15. Build canonical unified records + mapping report + license manifest
& .venv\Scripts\python.exe src\data\prepare_unified.py
```

Outputs (all gitignored, regenerable):

- `data/processed/unified_records.parquet` — 38,540 rows: HaluEval 20,000
  (10,000 groups, grouped 70/15/15 split), RAGTruth 17,790 (2,965 `source_id`
  groups, spans/quality/task/model/split preserved), FaithBench 750
  (15 batches, worst-severity label mapping).
- `artifacts/results/dataset_mapping_report.json` + `.csv` — counts, label
  distributions, span types, exclusions, FaithBench label sensitivity,
  raw-file SHA-256 hashes.
- `artifacts/results/dataset_license_manifest.json` — licenses, revisions,
  grouping rules, citations.

Determinism is guaranteed: rerunning step 15 produces byte-identical parquet
and an identical content fingerprint.

### Colab runtime: use L4 (22.5 GB VRAM / 54 GB RAM)

- **L4 is strongly recommended** — the free T4 can crash under the combined
  B2–B4 workload (B4 in particular needs headroom after B3's heavy run).
- Every heavy phase **checkpoints to `Drive/halurisc_cache/`** immediately
  after finishing (B2 models/results, B3 features + results + figures, B4
  artifacts). Restore cells (`7b.0`, `7d.5`, `7d.6`, `7g.0`) verify hashes
  before reuse, so a crashed session **resumes from Drive without redoing
  completed phases**: restart runtime → cells 1–5 → 5b → 6/6b → 7d → restore
  cells → continue from the crashed cell.
- All cached artifacts are CPU-portable (XGBoost trained with
  `HALU_XGB_DEVICE=cpu`; B4 calibrators are pure sklearn) — no cross-platform
  loading errors after download. Cell 7i (`verify_artifacts.py`) proves every
  artifact loads and predicts before packaging; run it again locally after
  unzipping.
- If you also run the optional legacy (course-run) cell 7, cell 7.0 restores its
  hash-matched root artifacts and cell 7 checkpoints them immediately to
  `halurisc_cache/version_a/`. Cell 8.0 restores the legacy analyses
  (cells 8–12, SHAP/RAGTruth/error/latency/LLM-judge) and cell 12.5
  checkpoints them; each legacy cell also skips when its outputs already
  exist. L4 tuning: cells 6 and 7e extract features with `--batch-size 512`.

### 2. Frontend Setup (Next.js + assistant-ui)

```powershell
# Open a second terminal window
cd web

# Install Node dependencies (pnpm — do not use npm)
pnpm install

# Configure environment variables
# Edit web/.env.local and add your OPENAI_API_KEY

# Run Next.js dev server (Port 3000)
pnpm run dev
```

Open `http://localhost:3000` in your browser.

---

## ☁️ Train on Colab (GPU, no local GPU needed)

**Self-contained notebook — upload ONE file only.** `colab/HaluRISC_Training_Version_B.ipynb`
embeds the entire source tree (29 files, base64) in cell 3. Running cell 3
writes everything to `/content/HaluRISC/`, verifies SHA-256 hashes, and `chdir`s
there — no source zip upload is needed. To patch a script later, edit
that file's `EMBEDDED` entry in cell 3 (or paste a small cell that rewrites the
file) and rerun cell 3. Regenerate the notebook after any source change with:

```powershell
& .venv\Scripts\python.exe colab\build_self_contained.py
```

Run the **full training pipeline** (feature extraction → XGBoost tuning → calibration → SHAP → RAGTruth validation) on Google Colab with a GPU, then download the artifacts back into this repo:

[![Open In Colab](https://colab.research.google.com/drive/124wjKFVDyZkDNIjs1WgHW8N7vO7XyY6G?usp=sharing)

1. Open the notebook (viewable by anyone with the link), select **GPU → L4** when available.
2. Upload only `HaluRISC_Training_Version_B.ipynb`; cell 3 embeds and verifies the runtime source.
3. Cell 35 saves `halurisc_artifacts_<date>.zip` to your Google Drive — download it and unzip **at the repo root**.
4. Start the API and web app as above — they automatically load the trained artifacts.

---

## 🔁 Reproducibility (B6)

Phase-specific guides live in **[`docs/`](docs/README.md)** — including the
B5.5 **manual reviewer guide** (`docs/b5-explanation-reliability.md`), the
B6 reproducibility protocol, and the B7 research-UI runbook.

The full pipeline (data → baselines → cross-domain → calibration shift → explanation reliability → manifest → verify) is driven by a single config:

```powershell
# Inspect the exact commands (no execution):
& .venv\Scripts\python.exe src\models\run_all_experiments.py --dry-run

# Run everything (CPU; use --device cuda on a GPU box):
& .venv\Scripts\python.exe src\models\run_all_experiments.py --device cpu

# Run only B3→B5 (e.g. after a crash), continuing past a failed phase:
& .venv\Scripts\python.exe src\models\run_all_experiments.py --from b3 --to b5 --keep-going
```

- Config: `configs/version_b.yaml` (schema `b6-run-all-v1`); each phase subprocess-invokes the existing per-phase runner, so the runners stay the single source of truth.
- Every run writes `artifacts/results/run_all_report.json` (resolved args, per-phase exit codes/durations, git commit).
- `artifacts/results/manifest.json` records processed + raw dataset hashes, split report, seeds `[42,123,456]`, feature groups, package versions, hardware, `HALU_*` env, and every produced artifact (internal checkpoints excluded). On Colab (no git) it also carries a `source_fingerprint` = sha256 over the notebook's embedded source hashes.

**CPU-compatible Docker path (backend only; CUDA stays an optional local acceleration path):**

```powershell
docker build -t halurisc-api .
docker run --rm -p 8000:8000 halurisc-api   # one worker, no --reload
curl http://127.0.0.1:8000/health
```

The image ships the exact pinned `requirements.txt` and `artifacts/` — a clean clone plus the documented Colab downloads (README "Train on Colab") regenerates everything without hidden local paths.

---

## 📁 Repository Structure

```
HaluRISC/
├── blueprint.md            # Unified research blueprint (single source of truth)
├── proposal.md             # Project proposal document
├── roadmap.md              # Implementation record (both phases complete)
├── AGENTS.md               # AI agent rules (prompt caching, boundaries, paper rules)
├── LICENSE                 # MIT License
├── .env.example            # Environment variables template
├── requirements.txt        # Pinned Python dependencies (exact pins)
├── Journal_Paper/          # Journal manuscript (halurisc.tex, CAS single column) + figures/screenshots
├── docs/                   # Phase guides (b5/b6/b7) + frozen manifest
├── data/
│   ├── raw/                # gitignored raw corpora + revision.json provenance
│   └── processed/          # clean parquet, unified records, audit samples
├── src/
│   ├── data/               # download.py, prepare.py, prepare_unified.py, registry.py
│   ├── features/           # extract_features.py + entity/nli/semantic modules
│   ├── models/             # run_all_experiments.py, train_pipeline.py, run_b2..b5, analyze_length_shortcut.py
│   ├── explain/            # shap_analysis.py
│   └── api/                # FastAPI main.py (/predict, /explain, /verify, /judge, /health)
├── configs/                # version_b.yaml (run-all protocol)
├── colab/                  # self-contained HaluRISC_Training_Version_B.ipynb + helpers
├── artifacts/
│   ├── models/             # model, calibrator, scaler, SHAP explainer artifacts
│   ├── results/            # b2..b5 metric tables + audit sheet + length analysis
│   └── split_indices.json  # Saved grouped 70/15/15 split indices
├── tests/                  # pytest suites (227 tests)
└── web/                    # Next.js App Router frontend
    ├── app/                # /chat, /analyze, /dashboard, /demo, /about, /api/chat
    ├── components/         # RiskGauge, ShapChart, NavBar, ThemeToggle, assistant-ui/thread
    └── toolkit.tsx         # Generative UI toolkit definition
```

---

## 📜 Dataset & Licensing Disclosures

- **HaluRISC Codebase**: Released under the **[MIT License](LICENSE)**.
- **HaluEval** (MIT): downloaded from the official [RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval) repository (EMNLP 2023), pinned to commit `b7253db3cdaa` with the file SHA-256 recorded in `data/raw/halueval/revision.json`.
- **RAGTruth** (MIT): external evaluation corpus from [ParticleMedia/RAGTruth](https://github.com/ParticleMedia/RAGTruth) (ACL 2024), pinned to commit `c103204b9ce2`.
- **FaithBench** (CC BY-NC-SA 4.0): stress-test corpus from [vectara/FaithBench](https://github.com/vectara/FaithBench) (NAACL 2025). Raw files are never redistributed; the repository ships the downloader, revision, and hashes only.
- No dataset is downloaded from Kaggle or an unverified mirror. Raw and restricted files stay gitignored; only scripts, hashes, citations, and license notes ship with the release.

---

## 👥 Authors

- **Saiful Alam Sabbir** (0112320105)
- **Md. Atikur Rahaman** (0112310298)
- **MD. Miraz Ahamed** (0112310524)

_Supervisor:_ **Ohidujjaman Tuhin, PhD**  
_Course:_ Machine Learning (Section E)
