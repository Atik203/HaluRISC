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
- ⚡ **Lightweight & Fast**: ~125 ms per analysis (p50 total over 200 test samples; model inference alone 4.5 ms).
- 💰 **~100x Cheaper than LLM Judges**: measured $0.101/1K predictions with GPT 5.6 Luna as judge vs near-zero local cost; also ~290x faster (1.29 s vs ~4.5 ms per sample).
- 🔬 **Statistically Rigorous**: 20,000 samples (HaluEval QA), grouped 70/15/15 splits (leakage-free), 3-seed protocol (42/123/456), grouped 5-fold CV tuning, McNemar + bootstrap CIs, calibration ECE raw 0.0122 / Platt 0.0101 / isotonic 0.0071 (corrected).

---

## 📊 Benchmark Results (HaluEval QA Holdout Test Set, N=3,000)

Corrected leakage-free grouped split; mean over seeds 42/123/456
(real results from `artifacts/results/b2/b2_model_comparison.json`):

| Model Architecture          | Precision  | Recall     | F1-Score   | AUROC      | PR-AUC     | MCC        |
| --------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **Majority (all 0)**        | 0.0000     | 0.0000     | 0.0000     | n/a        | n/a        | 0.0000     |
| **Heuristic (1 - overlap)** | 0.9473     | 0.9233     | 0.9352     | 0.9132     | 0.8196     | 0.8723     |
| **TF-IDF (Q+C+A)**          | 0.6403     | 0.5660     | 0.6008     | 0.6753     | 0.6785     | 0.2497     |
| **TF-IDF (answer only)**    | 0.9550     | 0.8920     | 0.9224     | 0.9696     | 0.9665     | 0.8519     |
| **TF-IDF (context only)**   | 0.5000     | 1.0000     | 0.6667     | n/a        | n/a        | 0.0000     |
| **NLI-only**                | 0.6479     | 0.6967     | 0.6714     | 0.7178     | 0.7533     | 0.3189     |
| **Logistic Regression**     | 0.9768     | 0.9553     | 0.9660     | 0.9932     | 0.9900     | 0.9329     |
| **Random Forest**           | 0.9890     | 0.9789     | 0.9839     | 0.9981     | 0.9984     | 0.9681     |
| **XGBoost (ours)**          | **0.9935** | **0.9780** | **0.9857** | **0.9980** | **0.9984** | **0.9717** |

**Artifact controls (B2):** answer-only TF-IDF already reaches F1 0.9224 — a
large part of the HaluEval signal is answer-style surface text. Context-only
has zero signal (paired answers share context, F1 0.6667 = always-positive).
NLI-only is weak (0.6714). The full evidence-aware model (XGBoost 0.9857)
adds a significant margin over all controls (McNemar p < 1e-5 vs answer-only).
Uncalibrated XGBoost ECE 0.0065 (formal calibration comparison: B4).

**Leakage-removal impact (B2):** historical row-level-split F1 0.9886 →
corrected grouped split + grouped-CV tuning F1 0.9857 (Δ −0.003; AUROC
unchanged at 0.9980). The corrected Version A reference (row-CV tuning) was
F1 0.9842. Details: `artifacts/results/b2/b2_leakage_comparison.json`.

### LLM-as-Judge comparison (200 test samples, measured)

| Model              | Accuracy   | Precision  | Recall     | F1         | Latency p50 | Cost / 1K |
| ------------------ | ---------- | ---------- | ---------- | ---------- | ----------- | --------- |
| GPT 5.6 Luna judge | 0.8600     | 0.9737     | 0.7400     | 0.8409     | 1,310 ms    | $0.105    |
| **XGBoost (ours)** | **0.9850** | **0.9802** | **0.9900** | **0.9851** | ~5 ms       | ~$0.001   |

Agreement between judge and XGBoost: 0.845 (McNemar p = 1.6e-05).

### External zero-shot validation (RAGTruth QA, 2,000 samples, no training)

F1 0.4819 · AUROC 0.5797 · ECE 0.6651 — the HaluEval-trained model does
**not** transfer to natural RAG responses (recall 1.0 = flags almost everything
risky). This is an honest finding: synthetic HaluEval patterns differ from
real-world generation, motivating cross-domain adaptation (Version B B3).

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

- Windows 11, Python 3.12.13, NVIDIA RTX 3060 6GB (CUDA 12.8, torch 2.11.0+cu128), 32 GB RAM
- scikit-learn 1.9.0, xgboost 3.4.0, shap 0.52.0, spacy 3.8.14 (en-core-web-sm 3.8.0), sentence-transformers 5.6.1

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

### Version B — Unified dataset layer (B1)

Canonical, lossless schema for HaluEval + RAGTruth (official) + FaithBench
(roadmap §14 B1). Version A preprocessing (`prepare.py`) is untouched; B1 adds
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

Run the **full training pipeline** (feature extraction → XGBoost tuning → calibration → SHAP → RAGTruth validation) on Google Colab with a GPU, then download the artifacts back into this repo:

[![Open In Colab](https://colab.research.google.com/drive/124wjKFVDyZkDNIjs1WgHW8N7vO7XyY6G?usp=sharing)

1. Open the notebook (viewable by anyone with the link), select **GPU → T4** as the runtime.
2. Run cells in order; cell 3 prompts for `colab/halurisc_src.zip` (from this repo).
3. Cell 11 saves `halurisc_artifacts_<date>.zip` to your Google Drive — download it and unzip **at the repo root**.
4. Start the API and web app as above — they automatically load the trained artifacts.

---

## 📁 Repository Structure

```
HaluRISC/
├── blueprint.md            # Research blueprint & two-version study plan
├── proposal.md             # Project proposal document
├── roadmap.md              # Detailed 8-week implementation roadmap
├── AGENTS.md               # AI Agent optimization guidelines & prompt caching protocol
├── LICENSE                 # MIT License
├── .env.example            # Environment variables template
├── requirements.txt        # Pinned Python dependencies
├── data/
│   ├── raw/halueval/       # Downloaded raw qa_data.json
│   └── processed/          # Clean parquet dataset, audit 50 samples
├── src/
│   ├── data/               # download.py, prepare.py, download_ragtruth.py
│   ├── features/           # extract_features.py + entity/nli/semantic modules
│   ├── models/             # train_pipeline.py, config.py, error_analysis.py, eval_llm_judge.py, eval_efficiency.py
│   ├── explain/            # shap_analysis.py
│   └── api/                # FastAPI main.py (/predict, /explain, /judge, /health)
├── colab/                  # HaluRISC_Training.ipynb + halurisc_src.zip bundle
├── artifacts/
│   ├── models/             # Model artifacts and params
│   ├── results/            # baseline_results.csv, JSON outputs
│   └── split_indices.json  # Saved 70/15/15 split indices
└── web/                    # Next.js App Router frontend
    ├── app/                # /chat, /analyze, /dashboard, /about, /api/chat
    ├── components/         # RiskGauge, ShapChart, NavBar, ThemeToggle, assistant-ui/thread
    └── toolkit.tsx         # Generative UI toolkit definition
```

---

## 📜 Dataset & Licensing Disclosures

- **HaluRISC Codebase**: Released under the **[MIT License](LICENSE)**.
- **HaluEval Benchmark Dataset**: Downloaded from the official [RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval) repository (EMNLP 2023). Note: HaluEval dataset files are used locally for academic research evaluation purposes.
- **RAGTruth Dataset**: External holdout evaluation corpus from [ParticleMedia/RAGTruth](https://github.com/ParticleMedia/RAGTruth) (ACL 2024).

---

## 👥 Authors

- **Saiful Alam Sabbir** (0112320105)
- **Md. Atikur Rahaman** (0112310298)
- **MD. Miraz Ahamed** (0112310524)

_Supervisor:_ **Ohidujjaman Tuhin, PhD**  
_Course:_ Machine Learning (Section E)
