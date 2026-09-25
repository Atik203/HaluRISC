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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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

THRESHOLDS = {"low": 0.35, "medium": 0.60, "high": 1.0}
WARNING = ("Calibrated on natural RAGTruth responses. The model itself is trained on "
           "HaluEval synthetic data, so the score is style-sensitive; per-claim "
           "verdicts lead when present.")

# B-run deployable (B4.2 predeclared rule): B2 xgboost_seed_42 + B4 Platt
# source calibrator. Falls back to the Version A bundle when B-run artifacts
# are missing (e.g. a VA-only clone).
B2_MODEL = MODELS_DIR / "b2" / "xgboost_seed_42.joblib"
B4_PLATT = MODELS_DIR / "b4" / "calibrator_platt_source_seed_42.joblib"
B4_DISPLAY = MODELS_DIR / "b4" / "calibrator_display.joblib"

# B2 comparison baselines for /predict/compare (seed 42, serving only).
B2_RF = MODELS_DIR / "b2" / "random_forest_seed_42.joblib"
B2_LR = MODELS_DIR / "b2" / "logistic_regression_full_seed_42.joblib"
B2_SCALER = MODELS_DIR / "b2" / "scaler_full.joblib"
HEURISTIC_OVERLAP_THRESHOLD = 0.97

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

STATE = {"model": None, "explainer": None, "feature_models": None, "feature_cols": None,
         "params": None, "baselines": {}}

# T3: lazy retrieval singletons (document index + Brave/Tavily web search).
RETRIEVAL_LOCK = threading.Lock()
RETRIEVAL_INDEX = None
WEB_SEARCH = None
BRAVE_ANSWERS = None
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = 25 * 1024 * 1024


def _embed_texts(texts: List[str]) -> np.ndarray:
    """SBERT embeddings from the shared feature models (used by the index)."""
    try:
        models = load_feature_models()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Feature models unavailable: {e}")
    embedder = models.get("embedder")
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded.")
    return np.asarray(embedder.encode(texts), dtype="float32")


def get_retrieval_index():
    """Lazy singleton document index (disk-persisted)."""
    global RETRIEVAL_INDEX
    if RETRIEVAL_INDEX is None:
        with RETRIEVAL_LOCK:
            if RETRIEVAL_INDEX is None:
                from src.retrieval import RetrievalIndex

                RETRIEVAL_INDEX = RetrievalIndex(embed_fn=_embed_texts)
    return RETRIEVAL_INDEX


def get_web_search():
    """Lazy singleton web search (Brave preferred, Tavily fallback; root .env)."""
    global WEB_SEARCH
    if WEB_SEARCH is None:
        with RETRIEVAL_LOCK:
            if WEB_SEARCH is None:
                from src.retrieval import WebSearch

                WEB_SEARCH = WebSearch()
    return WEB_SEARCH


def get_brave_answers():
    """Lazy singleton Brave Answers client (optional /answer endpoint)."""
    global BRAVE_ANSWERS
    if BRAVE_ANSWERS is None:
        with RETRIEVAL_LOCK:
            if BRAVE_ANSWERS is None:
                from src.retrieval import BraveAnswers

                BRAVE_ANSWERS = BraveAnswers()
    return BRAVE_ANSWERS


def _maybe_rerank(query: str, candidates: list, top_k: int) -> list:
    from src.retrieval import rerank

    return rerank.rerank(query, candidates, top_k)


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
    legacy_score: float
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


class CompareModelScore(BaseModel):
    """One B2 model's score for the same input (raw probability)."""
    score: float
    label: str
    decision_threshold: float


class DeployedComparison(BaseModel):
    """The deployed score triple from /predict, kept for reference."""
    calibrated_score: float
    risk_score: float
    legacy_score: float
    label: str


class CompareResponse(BaseModel):
    """Same input scored by every served model (additive endpoint)."""
    models: Dict[str, CompareModelScore]
    deployed: DeployedComparison
    thresholds: Dict[str, float]
    latency_ms: float
    model_version: str
    feature_version: str
    warning: str


class ClaimVerdict(BaseModel):
    """B7.5 Tier 2-4: one atomic claim with its NLI/LLM verdict."""
    id: int
    text: str
    verdict: str  # supported | contradicted | unsupported
    confidence: float
    evidence_sentence: str
    evidence_quote: str = ""    # exact contradicting sentence (corrective snippet)
    evidence_source: str = ""   # "context" | "doc:<name>" | "web:<url>"
    evidence_url: str = ""
    abstained: bool = False
    judged_by: str = "nli"      # nli | llm (Tier 4 hybrid routing)
    judge_reasoning: str = ""


class VerifyRequest(AnalysisRequest):
    """Tier 3/4: evidence selection + optional LLM-judge routing."""
    evidence_mode: Literal["auto", "context", "index", "web"] = "auto"
    judge_uncertain: bool = True


class VerifyResponse(BaseModel):
    """B7.5 Tier 2/3: claim-level verification + calibrated prediction.

    prediction/explanation are the Tier-1 secondary signals; claims carry the
    primary NLI-based verdicts with citations (Tier 3).
    """
    claims: List[ClaimVerdict]
    aggregate: Dict[str, object]
    prediction: PredictionResponse
    explanation: Optional[ExplanationResponse] = None


class RetrieveRequest(BaseModel):
    query: str = Field(..., max_length=2000)
    top_k: int = Field(5, ge=1, le=20)


class RetrievedPassage(BaseModel):
    id: str
    source: str
    url: str = ""
    text: str
    score: float = 0.0


class RetrieveResponse(BaseModel):
    evidence_mode: str
    passages: List[RetrievedPassage]


class IndexStatusResponse(BaseModel):
    n_passages: int
    n_documents: int
    dim: Optional[int]
    index_dir: str


class FeedbackRequest(BaseModel):
    """T4: human feedback on a verdict (aggregate or per-claim)."""
    inputs_hash: str = ""
    question: str = ""
    context: str = ""
    answer: str = ""
    claim_text: str = ""        # empty = aggregate feedback
    verdict: str = ""
    evidence_sentence: str = ""
    feedback: Literal["agree", "disagree"] = "agree"
    note: str = ""


class AnswerRequest(BaseModel):
    """Brave Answers query (grounded, cited reply; requires the Answers plan)."""
    question: str = Field(..., max_length=2000)
    country: str = Field("us", max_length=3)
    language: str = Field("en", max_length=8)
    research: bool = False


class AnswerCitation(BaseModel):
    number: Optional[int] = None
    url: str = ""
    snippet: str = ""
    start_index: Optional[int] = None
    end_index: Optional[int] = None


class AnswerResponse(BaseModel):
    answer: str
    citations: List[AnswerCitation]
    usage: Dict[str, object]
    model: str
    latency_ms: float


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
    """Deployable: B2 xgboost_seed_42 + B4 display calibrator (fitted on
    natural RAGTruth QA data) for calibrated_score, with the B4 source Platt
    (HaluEval val) kept as legacy_score. Falls back to the Version A
    xgb+platt bundle when B-run artifacts are missing."""
    import joblib

    def _legacy_proba(raw, platt, X):
        p = raw.predict_proba(X)[:, 1]
        return platt.predict_proba(p.reshape(-1, 1))

    if B2_MODEL.exists() and B4_PLATT.exists():
        raw = joblib.load(B2_MODEL)
        platt = joblib.load(B4_PLATT)
        display_bundle = joblib.load(B4_DISPLAY) if B4_DISPLAY.exists() else None

        def predict_proba(X):
            return _legacy_proba(raw, platt, X)

        def predict_proba_display(X):
            p_raw = raw.predict_proba(X)[:, 1]
            if display_bundle is None:
                return platt.predict_proba(p_raw.reshape(-1, 1))
            method = display_bundle["method"]
            cal = display_bundle["calibrator"]
            if method == "isotonic":
                res = cal.predict(p_raw)
            else:
                p_clip = np.clip(p_raw, 1e-12, 1 - 1e-12)
                z = np.log(p_clip / (1 - p_clip))
                if method == "temperature":
                    res = 1.0 / (1.0 + np.exp(-z / float(cal)))
                else:
                    res = cal.predict_proba(z.reshape(-1, 1))[:, 1]
            pos = np.asarray(res).reshape(-1, 1)
            return np.hstack([1.0 - pos, pos])

        logger.info("Deployable model: B2 xgboost_seed_42 + B4 display calibrator "
                    f"({display_bundle['method'] if display_bundle else 'source platt fallback'})")
        return {"raw": raw, "predict_proba": predict_proba,
                "predict_proba_display": predict_proba_display}

    logger.warning("B-run deployable artifacts missing; falling back to the Version A bundle")
    bundle = joblib.load(MODELS_DIR / "model_xgboost_calibrated.joblib")
    if isinstance(bundle, dict) and bundle.get("kind") == "xgb+platt":
        raw, platt = bundle["model"], bundle["calibrator"]

        def predict_proba(X):
            return _legacy_proba(raw, platt, X)

        return {"raw": raw, "predict_proba": predict_proba,
                "predict_proba_display": predict_proba}
    return {"raw": bundle, "predict_proba": lambda X: bundle.predict_proba(X),
            "predict_proba_display": lambda X: bundle.predict_proba(X)}


def _load_baseline_models() -> dict:
    """B2 seed-42 comparison baselines for /predict/compare (load only, never train).

    Only models with saved artifacts are served: random forest and logistic
    regression (with its StandardScaler). NLI-only and TF-IDF control models
    were not exported during B2, so they appear in the paper tables only.
    """
    import joblib

    baselines: dict = {}
    try:
        if B2_RF.exists():
            baselines["random_forest"] = {"model": joblib.load(B2_RF)}
        if B2_LR.exists() and B2_SCALER.exists():
            baselines["logistic_regression"] = {
                "model": joblib.load(B2_LR),
                "scaler": joblib.load(B2_SCALER),
            }
    except Exception as e:
        logger.warning(f"Baseline comparison models not loaded: {e}")
        return {}
    if baselines:
        logger.info(f"Baseline comparison models loaded: {sorted(baselines)}")
    return baselines


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
    STATE["baselines"] = _load_baseline_models()

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
    STATE.update({"model": None, "explainer": None, "feature_models": None,
                  "feature_cols": None, "params": None, "baselines": {}})  # reset schema, not clear()


# T4: per-IP rate limits (slowapi; env-tunable).
RATE_LIMIT_VERIFY = os.environ.get("HALU_RATE_VERIFY", "30/minute")
RATE_LIMIT_INDEX = os.environ.get("HALU_RATE_INDEX", "10/minute")
RATE_LIMIT_JUDGE = os.environ.get("HALU_RATE_JUDGE", "10/minute")
RATE_LIMIT_FEEDBACK = os.environ.get("HALU_RATE_FEEDBACK", "30/minute")
RATE_LIMIT_ANSWER = os.environ.get("HALU_RATE_ANSWER", "5/minute")

limiter = Limiter(key_func=get_remote_address, default_limits=[])

app = FastAPI(
    title="HaluRISC API",
    description="Calibrated & explainable hallucination-risk estimation (B-run deployable: B2 XGBoost + B4 Platt)",
    version="1.2.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    if p >= THRESHOLDS["medium"]:
        return "high_risk"
    if p >= THRESHOLDS["low"]:
        return "medium_risk"
    return "low_risk"


FEEDBACK_LOG = ROOT / "data" / "processed" / "feedback_log.jsonl"


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
        "baselines_loaded": sorted((STATE.get("baselines") or {}).keys()),
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
        "web_search": _web_search_meta(),
    }


def _web_search_meta() -> dict:
    """Provider info for the frontend (never raises)."""
    try:
        search = get_web_search()
        return {"provider": search.provider, "enabled": search.enabled,
                "answers_enabled": get_brave_answers().enabled}
    except Exception:
        return {"provider": None, "enabled": False, "answers_enabled": False}


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(req: AnalysisRequest):
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Run training (colab/HaluRISC_Training_Version_B.ipynb) and place artifacts/ in the repo root.")
    if not req.answer.strip():
        raise HTTPException(status_code=400, detail="Answer string cannot be empty.")

    t0 = time.time()
    feats = _feature_vector(req)
    X = np.array([[feats[c] for c in STATE["feature_cols"]]], dtype=np.float64)

    p_raw = float(STATE["model"]["raw"].predict_proba(X)[0, 1])
    p_raw = min(0.999, max(0.001, p_raw))
    p_disp = float(STATE["model"]["predict_proba_display"](X)[0, 1])
    p_disp = min(0.999, max(0.001, p_disp))
    p_leg = float(STATE["model"]["predict_proba"](X)[0, 1])
    p_leg = min(0.999, max(0.001, p_leg))

    latency = round((time.time() - t0) * 1000, 2)

    return PredictionResponse(
        risk_score=round(p_raw, 4),
        calibrated_score=round(p_disp, 4),
        legacy_score=round(p_leg, 4),
        label=_risk_label(p_disp),
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


@app.post("/predict/compare", response_model=CompareResponse)
def predict_compare(req: AnalysisRequest):
    """Score the same input with the B2 comparison baselines.

    Every served model uses the raw probability from its saved B2 artifact.
    The decision_threshold field reports the paper's decision rule (0.5 for
    the learned models, 1 - 0.97 for the overlap heuristic), while label uses
    the deployed display bands. The deployed calibrated/legacy/raw triple is
    returned separately. Additive endpoint; /predict is unchanged.
    """
    pred = predict_risk(req)
    feats = pred.features
    X = np.array([[feats[c] for c in STATE["feature_cols"]]], dtype=np.float64)

    models: Dict[str, CompareModelScore] = {
        "xgboost": CompareModelScore(score=pred.risk_score,
                                     label=_risk_label(pred.risk_score),
                                     decision_threshold=0.5),
    }
    for name, entry in (STATE.get("baselines") or {}).items():
        scaler = entry.get("scaler")
        Xm = scaler.transform(X) if scaler is not None else X
        p = float(entry["model"].predict_proba(Xm)[0, 1])
        p = min(0.999, max(0.001, p))
        models[name] = CompareModelScore(score=round(p, 4), label=_risk_label(p),
                                         decision_threshold=0.5)

    heuristic_score = round(min(0.999, max(0.001, 1.0 - float(feats.get("overlap_answer_context", 0.0)))), 4)
    models["heuristic_overlap"] = CompareModelScore(
        score=heuristic_score,
        label=_risk_label(heuristic_score),
        decision_threshold=round(1.0 - HEURISTIC_OVERLAP_THRESHOLD, 4),
    )

    return CompareResponse(
        models=models,
        deployed=DeployedComparison(
            calibrated_score=pred.calibrated_score,
            risk_score=pred.risk_score,
            legacy_score=pred.legacy_score,
            label=pred.label,
        ),
        thresholds=THRESHOLDS,
        latency_ms=pred.latency_ms,
        model_version=MODEL_VERSION,
        feature_version=FEATURE_VERSION,
        warning=WARNING,
    )


@app.get("/index", response_model=IndexStatusResponse)
def index_status():
    """T3: document index status (passages, documents, embedding dim)."""
    return get_retrieval_index().status()


@app.post("/index")
@limiter.limit(RATE_LIMIT_INDEX)
async def index_upload(request: Request, files: List[UploadFile] = File(...)):
    """T3: upload PDF/DOCX/TXT documents into the retrieval index."""
    from src.retrieval.chunk import extract_text_from_bytes

    documents = []
    total = 0
    for f in files:
        data = await f.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"{f.filename} exceeds {MAX_UPLOAD_BYTES // 2**20} MB")
        total += len(data)
        if total > MAX_TOTAL_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="total upload exceeds 25 MB")
        try:
            text = extract_text_from_bytes(f.filename or "upload.txt", data)
        except Exception:
            raise HTTPException(status_code=400, detail=f"could not parse {f.filename}")
        if not text.strip():
            raise HTTPException(status_code=400, detail=f"{f.filename} contains no extractable text")
        documents.append({"source": f"doc:{f.filename}", "text": text})

    result = get_retrieval_index().add_documents(documents)
    return {"indexed": result}


@app.delete("/index")
@limiter.limit(RATE_LIMIT_INDEX)
def index_clear(request: Request):
    """T3: wipe the document index."""
    get_retrieval_index().clear()
    return {"cleared": True}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_passages(req: RetrieveRequest):
    """T3: hybrid retrieval over the document index (BM25 + dense + rerank)."""
    index = get_retrieval_index()
    if index.status()["n_passages"] == 0:
        return RetrieveResponse(evidence_mode="index", passages=[])
    passages = index.search(req.query, top_k=req.top_k, rerank_fn=_maybe_rerank)
    return RetrieveResponse(
        evidence_mode="index",
        passages=[RetrievedPassage(**{k: p[k] for k in ("id", "source", "url", "text", "score")}) for p in passages],
    )


def _passages_for_claims(claims: List[str], mode: str):
    """Tier 3 evidence selection -> (evidence_mode, passages_by_claim | None).

    None = use the pasted context; otherwise a per-claim passage list.
    """
    if mode == "context":
        return "context", None
    if mode in ("auto", "index"):
        index = get_retrieval_index()
        if index.status()["n_passages"] > 0:
            per_claim = [index.search(c, top_k=3, rerank_fn=_maybe_rerank) for c in claims]
            return "index", per_claim
        if mode == "index":
            return "index", [[] for _ in claims]  # abstain everything
    if mode in ("auto", "web"):
        web = get_web_search()
        if web.enabled:
            per_claim = [web.search(c) for c in claims]
            # rerank web passages too (relevance; lazy cross-encoder, offline fallback)
            per_claim = [_maybe_rerank(c, ps, 3) if ps else ps for c, ps in zip(claims, per_claim)]
            return "web", per_claim
    return "context", None


@app.post("/verify", response_model=VerifyResponse)
@limiter.limit(RATE_LIMIT_VERIFY)
def verify_claims_endpoint(request: Request, req: VerifyRequest):
    """B7.5 Tier 2/3: per-claim NLI verification against evidence.

    evidence_mode: auto (index -> web -> context), context (pasted text),
    index (documents only), web (Tavily only). Claims without retrieved
    evidence abstain (unsupported, abstained=True).
    """
    from src.claims.decompose import split_claims
    from src.claims.verify import verify_claims as run_verify_context
    from src.claims.verify import verify_claims_against_passages

    claims = split_claims(req.answer or "")
    if not claims:
        raise HTTPException(status_code=400, detail="Answer contains no extractable claims.")

    try:
        nli_model = load_feature_models().get("nli")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"NLI models unavailable: {e}")
    if nli_model is None:
        raise HTTPException(status_code=503, detail="NLI model not loaded.")

    evidence_mode, passages_by_claim = _passages_for_claims(claims, req.evidence_mode)
    with INFERENCE_LOCK:  # CUDA-safe: NLI batching serialized like feature extraction
        if passages_by_claim is None:
            from src.claims.decompose import split_sentences

            result = run_verify_context(claims, req.context or "", nli_model)
            for c in result["claims"]:
                c["evidence_source"] = "context"
            result["aggregate"]["evidence_mode"] = "context"
            ctx_sents = split_sentences(req.context or "") or [req.context or ""]
            judged_passages = [[{"text": s, "source": "context", "url": ""} for s in ctx_sents]
                               for _ in result["claims"]]
        else:
            result = verify_claims_against_passages(claims, passages_by_claim, nli_model)
            result["aggregate"]["evidence_mode"] = evidence_mode
            judged_passages = passages_by_claim

    if req.judge_uncertain:
        from src.claims import judge as claim_judge
        from src.claims.verify import _aggregate

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            judge_call = claim_judge.openai_judge_call(api_key)
            result["claims"] = claim_judge.judge_claims(result["claims"], judged_passages, judge_call)
        result["aggregate"]["llm_judged"] = sum(
            1 for c in result["claims"] if c.get("judged_by") == "llm")
        abstained = result["aggregate"].get("abstained", 0)
        result["aggregate"].update(_aggregate(result["claims"]))
        result["aggregate"]["abstained"] = abstained

    prediction = predict_risk(req)
    try:
        explanation = explain_risk(req)
    except HTTPException as e:
        if e.status_code != 503:
            raise
        explanation = None

    # Evidence-adjusted display score: on natural inputs the model alone
    # cannot discriminate (RAGTruth QA AUROC 0.54), so the per-claim NLI
    # verdicts carry the strongest signal. Lift the score for contradicted
    # claims, lower it when everything is supported.
    verdicts = [c.get("verdict") for c in result["claims"]]
    if verdicts:
        n = len(verdicts)
        c_frac = sum(1 for v in verdicts if v == "contradicted") / n
        u_frac = sum(1 for v in verdicts if v == "unsupported") / n
        s_frac = sum(1 for v in verdicts if v == "supported") / n
        base = prediction.calibrated_score
        adj = base + 0.35 * c_frac + 0.15 * u_frac - 0.15 * s_frac
        adj = round(min(0.98, max(0.02, adj)), 4)
        prediction = prediction.model_copy(update={
            "calibrated_score": adj,
            "label": _risk_label(adj),
        })
        result["aggregate"]["model_calibrated_score"] = base
        result["aggregate"]["evidence_calibrated_score"] = adj
        result["aggregate"]["score_adjusted_by_claims"] = True

    return VerifyResponse(
        claims=[ClaimVerdict(**c) for c in result["claims"]],
        aggregate=result["aggregate"],
        prediction=prediction,
        explanation=explanation,
    )


@app.post("/feedback")
@limiter.limit(RATE_LIMIT_FEEDBACK)
def submit_feedback(request: Request, req: FeedbackRequest):
    """T4: append feedback to data/processed/feedback_log.jsonl (gitignored)."""
    row = {
        **req.model_dump(),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": MODEL_VERSION,
    }
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return {"ok": True, "logged": "feedback_log.jsonl"}


@app.post("/judge", response_model=JudgeResponse)
@limiter.limit(RATE_LIMIT_JUDGE)
def judge_answer(request: Request, req: JudgeRequest):
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


@app.post("/answer", response_model=AnswerResponse)
@limiter.limit(RATE_LIMIT_ANSWER)
def brave_answer(request: Request, req: AnswerRequest):
    """Brave Answers: a grounded, cited reply for one question.

    Separate from /verify on purpose. The reply is model-generated, so it must
    not enter the claim-verification evidence chain (that would make the
    verification circular). Costly endpoint (searches + tokens), so it has its
    own tighter rate limit.
    """
    service = get_brave_answers()
    if not service.enabled:
        raise HTTPException(status_code=503, detail="BRAVE_ANSWERS_API_KEY not configured in .env")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question string cannot be empty.")
    t0 = time.time()
    try:
        result = service.ask(req.question, country=req.country, language=req.language, research=req.research)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Brave Answers failed: {e}")
    latency = round((time.time() - t0) * 1000, 2)
    citations = [
        AnswerCitation(**(c if isinstance(c, dict) else {}))
        for c in (result.get("citations") or [])
        if isinstance(c, dict)
    ]
    return AnswerResponse(
        answer=result.get("answer", ""),
        citations=citations,
        usage=result.get("usage") or {},
        model=result.get("model", "brave"),
        latency_ms=latency,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("FASTAPI_HOST", "127.0.0.1"), port=int(os.environ.get("FASTAPI_PORT", "8000")))
