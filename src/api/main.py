"""
HaluRISC FastAPI inference server.

Endpoints:
  GET  /health   -> status, model/feature versions, artifacts loaded
  POST /predict  -> calibrated hallucination-risk prediction for {question, context, answer}
  POST /explain  -> SHAP top-feature explanation for the same inputs
  POST /judge    -> LLM-as-judge (GPT 5.6 Luna) comparison baseline

Boundary rule: the API LOADS artifacts and feature models at startup; it NEVER trains.

Stability: heavy models (spaCy, NLI, SBERT) are preloaded at startup. Set
HALU_API_DEVICE=cpu (default) to avoid VRAM OOM / driver crashes on small GPUs;
HALU_API_DEVICE=cuda opts into GPU inference. Prefer running WITHOUT --reload
(uvicorn's file watcher can restart the server when repo files change).

Run (repo root, .venv):
  python -m uvicorn src.api.main:app --port 8000
"""

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Must be set before any CUDA context is created (model preload below).
# expandable_segments fights VRAM fragmentation on small GPUs (RTX 3060 6 GB);
# TOKENIZERS_PARALLELISM=false avoids tokenizer thread deadlocks on Windows.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("halurisc_api")

load_dotenv()  # root .env (FASTAPI_*, OPENAI_API_KEY, OPENAI_MODEL)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "artifacts" / "models"

MAX_ANSWER_CHARS = 20000
MAX_CONTEXT_CHARS = 20000

MODEL_VERSION = "b2-xgboost-v1.0"
FEATURE_VERSION = "course-v1.0"

THRESHOLDS = {"low": 0.30, "medium": 0.70, "high": 1.0}
WARNING = "Trained on HaluEval synthetic data. Results may not generalize to real-world LLM outputs."

# B-run deployable (B4.2 predeclared rule): B2 xgboost_seed_42 + B4 Platt
# source calibrator. Falls back to the Version A bundle when B-run artifacts
# are missing (e.g. a VA-only clone).
B2_MODEL = MODELS_DIR / "b2" / "xgboost_seed_42.joblib"
B4_PLATT = MODELS_DIR / "b4" / "calibrator_platt_source_seed_42.joblib"

# Heavy models (spaCy + NLI + SBERT) are preloaded at startup and loaded lazily
# on first request only if startup failed. The lock prevents concurrent
# double-loading, which previously caused memory spikes and process exits.
FEATURE_MODEL_LOAD_LOCK = threading.Lock()

# sentence-transformers is not fully thread-safe and concurrent CUDA inference
# from uvicorn's threadpool crashed the process (silent exit). All GPU feature
# extraction is serialized through this lock.
INFERENCE_LOCK = threading.Lock()

# Feature-vector LRU cache (roadmap B7.11: feature-result caching). Feature
# extraction is the slow part (NLI/embeddings); predicting/explaining the same
# inputs twice skips it. 256 entries of 26 floats is negligible memory.
FEATURE_CACHE: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
FEATURE_CACHE_MAX = 256

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


class AnalyzeResponse(BaseModel):
    """B7.5 Tier 1: combined predict + explain for auto-analysis cards."""
    prediction: PredictionResponse
    explanation: Optional[ExplanationResponse] = None


class ClaimVerdict(BaseModel):
    """B7.5 Tier 2: one atomic claim with its NLI-based verdict."""
    id: int
    text: str
    verdict: str  # supported | contradicted | unsupported
    confidence: float
    evidence_sentence: str


class VerifyResponse(BaseModel):
    """B7.5 Tier 2: claim-level verification + calibrated prediction.

    prediction/explanation are the Tier-1 secondary signals; claims carry the
    primary NLI-based verdicts.
    """
    claims: List[ClaimVerdict]
    aggregate: Dict[str, object]
    prediction: PredictionResponse
    explanation: Optional[ExplanationResponse] = None


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
    """Deployable: B2 xgboost_seed_42 + B4 Platt source calibrator (B-run);
    predict_proba = platt(raw.predict_proba). Falls back to the Version A
    xgb+platt bundle when B-run artifacts are missing."""
    import joblib

    if B2_MODEL.exists() and B4_PLATT.exists():
        raw = joblib.load(B2_MODEL)
        platt = joblib.load(B4_PLATT)

        def predict_proba(X):
            p = raw.predict_proba(X)[:, 1]
            return platt.predict_proba(p.reshape(-1, 1))

        logger.info("Deployable model: B2 xgboost_seed_42 + B4 Platt source calibrator")
        return {"raw": raw, "predict_proba": predict_proba}

    logger.warning("B-run deployable artifacts missing; falling back to the Version A bundle")
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

    b_run_ok = B2_MODEL.exists() and B4_PLATT.exists()
    va_ok = (MODELS_DIR / "model_xgboost_calibrated.joblib").exists()
    missing = [n for n in ["feature_names.json", "params.json"] if _missing(n)]
    if not (b_run_ok or va_ok):
        missing.append("b2/xgboost_seed_42.joblib + b4/calibrator_platt_source_seed_42.joblib "
                       "(or legacy model_xgboost_calibrated.joblib)")
    if missing:
        logger.warning(f"Missing artifacts: {missing} - run training first (colab/HaluRISC_Training_Version_B.ipynb)")
        return False

    import joblib

    STATE["model"] = _load_calibrated_model()
    STATE["params"] = json.loads((MODELS_DIR / "params.json").read_text())
    STATE["feature_cols"] = json.loads((MODELS_DIR / "feature_names.json").read_text())

    try:
        import joblib

        explainer_path = MODELS_DIR / "shap_explainer.joblib"
        if explainer_path.exists():
            STATE["explainer"] = joblib.load(explainer_path)
            logger.info("Loaded saved SHAP explainer")
        else:
            import shap

            raw = STATE["model"]["raw"]
            STATE["explainer"] = shap.TreeExplainer(raw)
            logger.info("Built SHAP explainer from raw model")
    except Exception as e:
        logger.warning(f"SHAP explainer not loaded: {e}")

    logger.info("Artifacts loaded.")
    return True


def load_feature_models():
    """Thread-safe lazy load of NER + NLI + embedding models.

    Eagerly preloaded at startup (see lifespan); this is a fallback that must
    never run concurrently from multiple request threads.
    """
    if STATE["feature_models"] is not None:
        return STATE["feature_models"]

    with FEATURE_MODEL_LOAD_LOCK:
        if STATE["feature_models"] is not None:
            return STATE["feature_models"]

        from src.features.extract_features import load_heavy_models

        device = os.environ.get("HALU_API_DEVICE", "cpu")
        t0 = time.time()
        try:
            STATE["feature_models"] = load_heavy_models(device=device)
            logger.info(f"Feature models loaded in {time.time() - t0:.1f}s (device={device})")
        except Exception as e:
            logger.error(f"Feature models failed to load (device={device}): {e}")
            raise
    return STATE["feature_models"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    if os.environ.get("HALU_API_PRELOAD", "1") != "0":
        try:
            load_feature_models()
            STATE["models_ready"] = True
            logger.info("All models preloaded. API ready.")
        except Exception as e:
            STATE["models_ready"] = False
            logger.warning(f"Preload of heavy feature models failed ({e}); API still serving "
                           f"/health and /judge, /predict will retry on first request.")
    else:
        logger.info("HALU_API_PRELOAD=0 -> heavy models will load lazily on first /predict.")
    yield
    STATE.clear()


app = FastAPI(
    title="HaluRISC API",
    description="Calibrated & explainable hallucination-risk estimation (B-run deployable: B2 XGBoost + B4 Platt)",
    version="1.1.0",
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

    key = hashlib.sha256(
        f"{req.question}\x1f{req.context}\x1f{req.answer}".encode("utf-8")
    ).hexdigest()

    with INFERENCE_LOCK:
        cached = FEATURE_CACHE.get(key)
        if cached is not None:
            FEATURE_CACHE.move_to_end(key)
            return cached
        try:
            models = load_feature_models()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Feature models unavailable: {e}")
        feats = extract_all_features_single(req.question or "", req.context or "", req.answer, models)
        missing = [c for c in STATE["feature_cols"] if c not in feats]
        if missing:
            raise HTTPException(status_code=500, detail=f"Feature extractor missing columns: {missing}")
        FEATURE_CACHE[key] = feats
        FEATURE_CACHE.move_to_end(key)
        if len(FEATURE_CACHE) > FEATURE_CACHE_MAX:
            FEATURE_CACHE.popitem(last=False)
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
        "status": "ok" if artifacts_ok and STATE["feature_models"] is not None else "degraded",
        "model": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "artifacts_loaded": artifacts_ok,
        "feature_models_ready": STATE["feature_models"] is not None,
        "explainer_ready": STATE["explainer"] is not None,
        "n_features": len(STATE["feature_cols"]) if STATE["feature_cols"] else 0,
        "device": _active_device(),
    }


def _active_device() -> str:
    """Resolved inference device: 'cuda' only when requested AND available."""
    if os.environ.get("HALU_API_DEVICE", "cpu").lower() == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"


@app.get("/meta")
def api_meta():
    """Frontend metadata (B7): thresholds, warning, feature groups, versions.

    Additive endpoint; the /predict and /explain contracts are unchanged.
    The frontend uses this to render thresholds/warning/grouped features
    without hardcoding them.
    """
    feature_groups = None
    try:
        from src.models.train_pipeline import FEATURE_GROUPS

        feature_groups = FEATURE_GROUPS
    except Exception:
        pass
    return {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "n_features": len(STATE["feature_cols"]) if STATE["feature_cols"] else 0,
        "thresholds": THRESHOLDS,
        "warning": WARNING,
        "device": _active_device(),
        "feature_groups": feature_groups,
        "features_available": STATE["model"] is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(req: AnalysisRequest):
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Run training (colab/HaluRISC_Training_Version_B.ipynb) and place artifacts/ in the repo root.")
    if not req.answer.strip():
        raise HTTPException(status_code=400, detail="Answer string cannot be empty.")

    t0 = time.time()
    feats = _feature_vector(req)
    X = np.array([[feats[c] for c in STATE["feature_cols"]]], dtype=np.float64)

    p = float(STATE["model"]["predict_proba"](X)[0, 1])
    p = min(0.999, max(0.001, p))

    latency = round((time.time() - t0) * 1000, 2)

    return PredictionResponse(
        risk_score=round(p, 4),
        calibrated_score=round(p, 4),
        label=_risk_label(p),
        thresholds=THRESHOLDS,
        latency_ms=latency,
        model_version=MODEL_VERSION,
        feature_version=FEATURE_VERSION,
        warning=WARNING,
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


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_risk(req: AnalysisRequest):
    """Combined predict + explain (additive; B7.5 Tier 1 auto-analysis cards).

    The feature vector is computed once (LRU-cached), so this is one extraction
    + one prediction + one SHAP pass. Explanations degrade gracefully to null
    when the SHAP explainer is unavailable.
    """
    prediction = predict_risk(req)
    try:
        explanation = explain_risk(req)
    except HTTPException as e:
        if e.status_code == 503:
            explanation = None
        else:
            raise
    return AnalyzeResponse(prediction=prediction, explanation=explanation)


@app.post("/verify", response_model=VerifyResponse)
def verify_claims_endpoint(req: AnalysisRequest):
    """B7.5 Tier 2: per-claim NLI verification against the evidence.

    Splits the answer into atomic claims, scores each against every evidence
    sentence with the loaded NLI cross-encoder, and returns 3-way verdicts
    (supported / contradicted / unsupported) plus the calibrated prediction
    as the labelled secondary signal.
    """
    from src.claims.decompose import split_claims
    from src.claims.verify import verify_claims as run_verify

    claims = split_claims(req.answer or "")
    if not claims:
        raise HTTPException(status_code=400, detail="Answer contains no extractable claims.")

    try:
        nli_model = load_feature_models().get("nli")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"NLI models unavailable: {e}")
    if nli_model is None:
        raise HTTPException(status_code=503, detail="NLI model not loaded.")

    result = run_verify(claims, req.context or "", nli_model)

    prediction = predict_risk(req)
    try:
        explanation = explain_risk(req)
    except HTTPException as e:
        if e.status_code != 503:
            raise
        explanation = None

    return VerifyResponse(
        claims=[ClaimVerdict(**c) for c in result["claims"]],
        aggregate=result["aggregate"],
        prediction=prediction,
        explanation=explanation,
    )


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
