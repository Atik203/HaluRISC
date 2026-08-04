"""
HaluRISC FastAPI inference server.

Endpoints:
  GET  /health   -> status, model/feature versions, artifacts loaded
  POST /predict  -> calibrated hallucination-risk prediction for {question, context, answer}
  POST /explain  -> SHAP top-feature explanation for the same inputs
  POST /judge    -> LLM-as-judge (GPT 5.6 Luna) comparison baseline

Boundary rule: the API LOADS artifacts and feature models at startup; it NEVER trains.

Run (repo root, .venv):
  python -m uvicorn src.api.main:app --reload --port 8000
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("halurisc_api")

load_dotenv()  # root .env (FASTAPI_*, OPENAI_API_KEY, OPENAI_MODEL)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "artifacts" / "models"

MAX_ANSWER_CHARS = 20000
MAX_CONTEXT_CHARS = 20000

MODEL_VERSION = "xgboost-v1.0"
FEATURE_VERSION = "course-v1.0"

STATE = {"model": None, "explainer": None, "feature_models": None, "feature_cols": None, "params": None}


# ----------------------------------------------------------------------------
# Schemas (stable API contract, see AGENTS.md §8)
# ----------------------------------------------------------------------------
class AnalysisRequest(BaseModel):
    question: str = Field("", max_length=5000, description="The question that was asked")
    context: Optional[str] = Field("", max_length=MAX_CONTEXT_CHARS, description="Reference context/evidence")
    answer: str = Field(..., max_length=MAX_ANSWER_CHARS, description="Candidate LLM answer to score")
    domain: Optional[str] = "qa"


class FeatureImpact(BaseModel):
    feature: str
    value: float
    impact: float


class PredictionResponse(BaseModel):
    risk_score: float
    calibrated_score: float
    label: str
    thresholds: Dict[str, float]
    latency_ms: float
    model_version: str
    feature_version: str
    warning: str
    features: Dict[str, float]


class ExplanationResponse(BaseModel):
    top_features: List[FeatureImpact]
    base_value: float


class JudgeRequest(BaseModel):
    question: str = ""
    context: Optional[str] = ""
    answer: str = Field(..., max_length=MAX_ANSWER_CHARS)


class JudgeResponse(BaseModel):
    judgment: str
    confidence: float
    reasoning: str
    model: str


# ----------------------------------------------------------------------------
# Startup / artifact loading
# ----------------------------------------------------------------------------
def _load_calibrated_model():
    """Load the xgb+platt artifact; predict_proba = platt(raw.predict_proba)."""
    import joblib

    bundle = joblib.load(MODELS_DIR / "model_xgboost_calibrated.joblib")
    if isinstance(bundle, dict) and bundle.get("kind") == "xgb+platt":
        raw, platt = bundle["model"], bundle["calibrator"]

        def predict_proba(X):
            p = raw.predict_proba(X)[:, 1]
            return platt.predict_proba(p.reshape(-1, 1))

        return {"raw": raw, "predict_proba": predict_proba}
    return {"raw": bundle, "predict_proba": lambda X: bundle.predict_proba(X)}


def load_artifacts():
    def _missing(name: str) -> bool:
        return not (MODELS_DIR / name).exists()

    missing = [n for n in ["model_xgboost_calibrated.joblib", "model_xgboost_raw.joblib", "feature_names.json", "params.json"] if _missing(n)]
    if missing:
        logger.warning(f"Missing artifacts: {missing} - run training first (colab/HaluRISC_Training.ipynb)")
        return False

    import joblib

    STATE["model"] = _load_calibrated_model()
    STATE["params"] = json.loads((MODELS_DIR / "params.json").read_text())
    STATE["feature_cols"] = json.loads((MODELS_DIR / "feature_names.json").read_text())

    try:
        import shap

        raw = STATE["model"]["raw"]
        STATE["explainer"] = shap.TreeExplainer(raw)
    except Exception as e:
        logger.warning(f"SHAP explainer not loaded: {e}")

    logger.info("Artifacts loaded.")
    return True


def load_feature_models():
    if STATE["feature_models"] is None:
        from src.features.extract_features import load_heavy_models

        t0 = time.time()
        STATE["feature_models"] = load_heavy_models()
        logger.info(f"Feature models loaded in {time.time() - t0:.1f}s")
    return STATE["feature_models"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    STATE.clear()


app = FastAPI(
    title="HaluRISC API",
    description="Calibrated & explainable hallucination-risk estimation (Version A)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _feature_vector(req: AnalysisRequest) -> Dict[str, float]:
    from src.features.extract_features import extract_all_features_single

    models = load_feature_models()
    feats = extract_all_features_single(req.question or "", req.context or "", req.answer, models)
    missing = [c for c in STATE["feature_cols"] if c not in feats]
    if missing:
        raise HTTPException(status_code=500, detail=f"Feature extractor missing columns: {missing}")
    return feats


def _risk_label(p: float) -> str:
    if p >= 0.70:
        return "high_risk"
    if p >= 0.30:
        return "medium_risk"
    return "low_risk"


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.get("/health")
def health_check():
    artifacts_ok = STATE["model"] is not None
    return {
        "status": "ok" if artifacts_ok else "degraded",
        "model": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "artifacts_loaded": artifacts_ok,
        "explainer_ready": STATE["explainer"] is not None,
        "n_features": len(STATE["feature_cols"]) if STATE["feature_cols"] else 0,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(req: AnalysisRequest):
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Run training (colab/HaluRISC_Training.ipynb) and place artifacts/ in the repo root.")
    if not req.answer.strip():
        raise HTTPException(status_code=400, detail="Answer string cannot be empty.")

    t0 = time.time()
    feats = _feature_vector(req)
    X = np.array([[feats[c] for c in STATE["feature_cols"]]], dtype=np.float64)

    p = float(STATE["model"]["predict_proba"](X)[0, 1])
    p = min(0.999, max(0.001, p))

    thresholds = {"low": 0.30, "medium": 0.70, "high": 1.0}
    latency = round((time.time() - t0) * 1000, 2)

    return PredictionResponse(
        risk_score=round(p, 4),
        calibrated_score=round(p, 4),
        label=_risk_label(p),
        thresholds=thresholds,
        latency_ms=latency,
        model_version=MODEL_VERSION,
        feature_version=FEATURE_VERSION,
        warning="Trained on HaluEval synthetic data. Results may not generalize to real-world LLM outputs.",
        features={k: float(v) for k, v in feats.items()},
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain_risk(req: AnalysisRequest):
    if STATE["model"] is None or STATE["explainer"] is None:
        raise HTTPException(status_code=503, detail="Explainer not loaded. Run training first.")

    feats = _feature_vector(req)
    X = np.array([[feats[c] for c in STATE["feature_cols"]]], dtype=np.float64)

    shap_values = STATE["explainer"].shap_values(X)[0]
    base_value = float(STATE["explainer"].expected_value)
    order = np.argsort(np.abs(shap_values))[::-1][:5]

    top_features = [
        FeatureImpact(
            feature=STATE["feature_cols"][i],
            value=round(float(X[0, i]), 6),
            impact=round(float(shap_values[i]), 6),
        )
        for i in order
    ]
    return ExplanationResponse(top_features=top_features, base_value=round(base_value, 6))


@app.post("/judge", response_model=JudgeResponse)
def judge_answer(req: JudgeRequest):
    """LLM-as-judge baseline (GPT 5.6 Luna). Uses OPENAI_API_KEY from .env."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured in .env")

    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(status_code=503, detail="openai package not installed")

    model_name = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
    client = OpenAI(api_key=api_key)

    system = (
        "You are an expert hallucination-judge. Given a question, a reference context, and an answer, "
        "decide whether the answer contains hallucinated content (unsupported, contradictory, or fabricated "
        "information relative to the context). Respond with JSON only: "
        '{"judgment": "hallucinated"|"grounded", "confidence": 0.0-1.0, "reasoning": "<short explanation>"}.'
    )
    user = (
        f"Question: {req.question}\n"
        f"Context: {req.context or '(none)'}\n"
        f"Answer: {req.answer}"
    )

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=250,
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content[content.find("{") : content.rfind("}") + 1])
        return JudgeResponse(
            judgment=data.get("judgment", "grounded"),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
            model=model_name,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM judge failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("FASTAPI_HOST", "127.0.0.1"), port=int(os.environ.get("FASTAPI_PORT", "8000")))
