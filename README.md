# HaluRISC — Calibrated & Explainable Hallucination Risk Analyzer

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-green.svg)](https://fastapi.tiangolo.com/)
[![assistant-ui](https://img.shields.io/badge/assistant--ui-0.7-violet.svg)](https://www.assistant-ui.com/)

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
- ⚡ **Lightweight & Fast**: CPU inference in ~12 milliseconds per response.
- 💰 **100x Cheaper than LLM Judges**: Evaluates 10,000 samples at near-zero incremental cost compared to ~$1.10 for pure LLM evaluation.
- 🔬 **Statistically Rigorous**: Evaluated on 20,000 samples (HaluEval QA dataset) using 70/15/15 stratified splits, McNemar's test, and Platt scaling probability calibration.

---

## 📊 Benchmark Results (HaluEval QA Holdout Test Set, N=3,000)

| Model Architecture              | Precision  | Recall     | F1-Score   | AUROC      | PR-AUC     | MCC        |
| ------------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **Heuristic (Overlap)**         | 0.9392     | 0.9467     | 0.9429     | 0.9148     | 0.8117     | 0.8854     |
| **Logistic Regression**         | 0.9789     | 0.9607     | 0.9697     | 0.9936     | 0.9934     | 0.9402     |
| **Random Forest**               | 0.9880     | 0.9840     | 0.9860     | 0.9976     | 0.9983     | 0.9720     |
| **XGBoost (Calibrated — Ours)** | **0.9899** | **0.9820** | **0.9859** | **0.9974** | **0.9982** | **0.9720** |
| **GPT 5.6 Luna Judge Baseline** | 0.9450     | 0.9500     | 0.9470     | 0.9520     | 0.9400     | 0.8900     |

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

- **Python 3.12+**
- **Node.js 18+ & npm**
- **OpenAI API Key** (for conversational chat & LLM judge baseline)

---

### 1. Backend Setup & Data Pipeline

```powershell
# 1. Clone the repository
git clone https://github.com/Atik203/HaluLens.git
cd HaluRISC

# 2. Activate virtual environment (or create one)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download HaluEval QA dataset (10,000 JSONL records)
python src/data/download.py

# 5. Process binary dataset (20,000 rows) & generate 70/15/15 splits
python src/data/prepare.py

# 6. Extract core features (14 features)
python src/features/extract_features.py

# 7. Train & evaluate baseline ML models
python src/models/train_baselines.py

# 8. Start FastAPI inference backend server (Port 8000)
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

---

### 2. Frontend Setup (Next.js + assistant-ui)

```powershell
# Open a second terminal window
cd web

# Install Node dependencies
npm install

# Configure environment variables
# Edit web/.env.local and add your OPENAI_API_KEY
cp .env.local.example .env.local   # or edit existing .env.local

# Run Next.js dev server (Port 3000)
npm run dev
```

Open `http://localhost:3000` in your browser.

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
│   ├── data/               # download.py, prepare.py
│   ├── features/           # extract_features.py
│   ├── models/             # train_baselines.py
│   └── api/                # FastAPI main.py (/predict, /explain, /health)
├── artifacts/
│   ├── models/             # Model artifacts and params
│   ├── results/            # baseline_results.csv, JSON outputs
│   └── split_indices.json  # Saved 70/15/15 split indices
└── web/                    # Next.js App Router frontend
    ├── app/                # /chat, /analyze, /dashboard, /about, /api/chat
    ├── components/         # RiskGauge, ShapChart, NavBar
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
