"""WP6: LLM-judge and encoder comparison at natural prevalence (branch ``rebuild``).

This script evaluates two judges on the RAGTruth QA test split (900 rows,
prevalence 0.178) without balancing the sample:

  * ``--backend openai``: GPT-5.6-Luna through the OpenAI chat API.
  * ``--backend qwen``: Qwen 3.5 4B through a local Ollama server.

It also publishes the comparison rows required by WP6:

  * fixed-hyperparameter EC-XGB (m3) on the same 900 rows, F1 and AUROC at a
    0.5 threshold and at a 5% FPR operating point chosen on the 5,034-row QA
    calibration split;
  * a length-only logistic baseline on log1p(n_words), fit on the same
    5,034 calibration rows and evaluated on the 900 test rows.

Encoder timing (RoBERTa-base, 200 rows, batch 8, fp16 on CUDA) is implemented
as ``--time-encoder``.

Usage (repo root, ``.venv``)::

    python src/models/rebuild_judge.py --backend openai --limit 30 --repeats 3 --concurrency 6 --out-tag pilot
    python src/models/rebuild_judge.py --backend openai --limit 0 --repeats 3 --concurrency 6 --out-tag full
    python src/models/rebuild_judge.py --backend qwen --limit 0 --repeats 3 --concurrency 1 --out-tag full
    python src/models/rebuild_judge.py --comparisons-only
    python src/models/rebuild_judge.py --time-encoder

Outputs live in ``artifacts/results/rebuild/judge/``.  Nothing in this script
writes to the frozen ``b2``, ``b4``, ``b6``, or ``review`` result directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.config import BOOTSTRAP_SEED, N_BOOTSTRAP

# Load the root .env explicitly.  The key is never printed or written to an
# artifact; only the model name and token counts are recorded.
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rebuild_judge")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

RESULTS_DIR = ROOT / "artifacts" / "results" / "rebuild" / "judge"
FIXED_HP_DIR = ROOT / "artifacts" / "results" / "rebuild" / "fixed_hp"
FIXED_HP_MODELS = ROOT / "artifacts" / "models" / "rebuild" / "fixed_hp"
PREDICTIONS_PATH = FIXED_HP_DIR / "fixed_hp_predictions.parquet"
FEATURE_NAMES_PATH = FIXED_HP_DIR / "b6_feature_names.json"
UNIFIED_PATH = ROOT / "data" / "processed" / "unified_records.parquet"
CALIBRATION_FEATURES_PATH = ROOT / "data" / "processed" / "calibration_features.parquet"
EXTERNAL_FEATURES_PATH = ROOT / "data" / "processed" / "external_features.parquet"

DATASET = "ragtruth_qa_test"
EXPECTED_N = 900
EXPECTED_PREVALENCE = 0.1778
DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_PILOT_N = 30
MAX_COMPLETION_TOKENS = int(os.environ.get("HALU_JUDGE_MAX_COMPLETION_TOKENS", "512"))
QWEN_MAX_PREDICT = int(os.environ.get("HALU_QWEN_MAX_PREDICT", "300"))
QWEN_URL = os.environ.get("HALU_QWEN_URL", "http://localhost:11434")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.5:4b")

# Assumed GPT-5.6-Luna prices.  AGENTS.md gives the input prices (0.20/M
# uncached, 0.02/M cached); the output price is the value already used by
# src/models/eval_llm_judge.py.  Every run records these values so a reader
# can recompute the cost with different prices.
PRICING = {
    "currency": "USD",
    "input_per_mtok": float(os.environ.get("HALU_JUDGE_INPUT_USD_PER_MTOK", "0.20")),
    "cached_input_per_mtok": float(os.environ.get("HALU_JUDGE_CACHED_INPUT_USD_PER_MTOK", "0.02")),
    "output_per_mtok": float(os.environ.get("HALU_JUDGE_OUTPUT_USD_PER_MTOK", "1.20")),
    "assumption": "GPT-5.6-Luna list prices assumed; override with HALU_JUDGE_*_USD_PER_MTOK.",
}

# ---------------------------------------------------------------------------
# Prompt (static prefix; identical for every row and every repeat)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a factual-consistency judge for a research evaluation of hallucination detection. In each request you receive one QUESTION, one CONTEXT, and one ANSWER. Your task is to decide whether the ANSWER is grounded in the CONTEXT or hallucinated with respect to it.

DEFINITIONS
- A checkable claim is a factual assertion in the ANSWER that can be compared with the CONTEXT. Examples include statements about entities, events, numbers, dates, quantities, locations, causes, and relationships. Opinions, greetings, hedges, and procedural filler are not checkable claims.
- GROUNDED means every substantive checkable claim in the ANSWER is supported by at least one passage in the CONTEXT. Support can be a direct restatement, a paraphrase, or a combination of passages. The ANSWER does not need to reuse the wording of the CONTEXT.
- HALLUCINATED means at least one substantive checkable claim in the ANSWER is contradicted by the CONTEXT, or is not supported by any passage in the CONTEXT. A single unsupported or contradicted detail is enough, even when the rest of the ANSWER is correct.

DECISION PROCEDURE
1. Read the QUESTION to understand what the ANSWER is responding to. The QUESTION is not evidence. A claim that merely repeats the QUESTION is not automatically supported.
2. Read the CONTEXT carefully. It may contain several passages, repeated text, web boilerplate, or irrelevant material. Any single passage may support a claim.
3. List the checkable claims in the ANSWER internally. Do not show this list in your output.
4. For each claim, ask whether the CONTEXT entails it under a reasonable reading. Paraphrase and ordinary inference are allowed. World knowledge is not allowed.
5. If every checkable claim is supported, output grounded. If any checkable claim is contradicted or unsupported, output hallucinated.
6. If the ANSWER contains no checkable factual claim, output grounded.

RULES AND EDGE CASES
- Use only the CONTEXT. A claim that is true in the real world but absent from the CONTEXT is unsupported.
- Do not penalize the ANSWER for omitting information that is present in the CONTEXT. Completeness is not required.
- Do not reward fluency, length, detail, or confidence. A long, polished answer with one unsupported detail is hallucinated. A short, plain answer that is fully supported is grounded.
- Treat numbers, dates, names, locations, and quantities as substantive. If the ANSWER reports a different number, date, name, location, or quantity from the CONTEXT, treat the claim as contradicted unless the CONTEXT itself is ambiguous.
- Approximate quantities are supported when the CONTEXT states the same approximate quantity or a range that contains it. Rounding is acceptable when the rounded value is consistent with the CONTEXT.
- A claim that combines supported facts into a relationship the CONTEXT does not state is unsupported. For example, if the CONTEXT says A happened before B and C happened before D, do not infer that A happened before D.
- If the CONTEXT explicitly says that information is unknown, unavailable, or disputed, an ANSWER that asserts a definite value is unsupported.
- If two passages conflict, a claim supported by one of them is supported. Do not require the CONTEXT to be internally consistent.
- If the ANSWER contradicts itself, judge each substantive claim against the CONTEXT. A self-contradictory answer normally contains at least one unsupported claim.
- If the ANSWER is a refusal, a request for clarification, or a statement that the CONTEXT does not contain the answer, treat it as having no checkable claim and output grounded.
- If the ANSWER repeats the QUESTION without adding a factual assertion, output grounded.
- Do not evaluate grammar, style, helpfulness, safety, or whether the ANSWER follows the requested format.

WORKED EXAMPLES
- CONTEXT: "The bridge opened in 1937 and carries four lanes." ANSWER: "The bridge opened in 1937 and carries six lanes." Judgment: hallucinated, because the lane count contradicts the CONTEXT.
- CONTEXT: "The study enrolled 120 adults and reported a 12 percent reduction." ANSWER: "The study enrolled 120 adults." Judgment: grounded, because the stated claim is supported and the omission is not penalized.
- CONTEXT: "The plant operates in Ohio." ANSWER: "The plant operates in Ohio and is the largest in the country." Judgment: hallucinated, because the size claim is not supported even if it is true in the real world.

CONFIDENCE
Report confidence as your probability that your own judgment is correct, not the probability that the ANSWER is hallucinated. Use values near 0.5 when the CONTEXT is ambiguous or the claim is borderline, values near 0.7 when the judgment is plausible but debatable, and values above 0.9 only when the CONTEXT clearly supports or clearly contradicts the claim. Do not use 1.0.

OUTPUT
Return one JSON object and nothing else. Do not use markdown fences, preamble, or trailing commentary. The object must have exactly these keys:
{"judgment": "grounded" or "hallucinated", "confidence": <number between 0 and 1>, "reason": "<at most 25 words naming the deciding claim>"}
The reason must name the claim that decided the judgment. If the judgment is grounded, name the main supported claim. Keep the reason under 25 words.
"""

USER_TEMPLATE = "QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"

PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def count_tokens(text: str) -> int:
    """Best-effort token count for the prompt documentation."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return int(round(len(text) / 4))


SYSTEM_PROMPT_TOKENS = count_tokens(SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_test_rows() -> pd.DataFrame:
    """Return the 900 RAGTruth QA test rows with their question/context/answer."""
    pred = pd.read_parquet(PREDICTIONS_PATH, columns=["dataset", "sample_id", "label"])
    ids = pred.loc[pred["dataset"] == DATASET, ["sample_id", "label"]].drop_duplicates("sample_id")
    if len(ids) != EXPECTED_N:
        raise RuntimeError(f"expected {EXPECTED_N} {DATASET} ids, found {len(ids)} in {PREDICTIONS_PATH}")

    unified = pd.read_parquet(UNIFIED_PATH, columns=["sample_id", "question", "context", "answer"])
    merged = ids.merge(unified, on="sample_id", how="left", validate="one_to_one")
    for col in ["question", "context", "answer"]:
        missing = merged[col].isna() | (merged[col].astype(str).str.strip() == "")
        if missing.any():
            bad = merged.loc[missing, "sample_id"].head(5).tolist()
            raise RuntimeError(
                f"text unavailable for {int(missing.sum())} rows in {UNIFIED_PATH} "
                f"(examples: {bad}); cannot run the judge"
            )
        merged[col] = merged[col].astype(str)

    prevalence = float(merged["label"].mean())
    logger.info("Loaded %d %s rows (prevalence %.4f)", len(merged), DATASET, prevalence)
    if abs(prevalence - EXPECTED_PREVALENCE) > 0.005:
        logger.warning("prevalence %.4f differs from the expected %.4f", prevalence, EXPECTED_PREVALENCE)
    return merged.reset_index(drop=True)


def select_pilot(rows: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified pilot sample that preserves the natural prevalence."""
    if n >= len(rows):
        return rows.reset_index(drop=True)
    prevalence = float(rows["label"].mean())
    n_pos = int(round(n * prevalence))
    n_pos = max(1, min(n_pos, n - 1, int(rows["label"].sum())))
    n_neg = n - n_pos
    rng = np.random.default_rng(seed)
    pos_idx = rng.choice(rows.index[rows["label"] == 1].to_numpy(), n_pos, replace=False)
    neg_idx = rng.choice(rows.index[rows["label"] == 0].to_numpy(), n_neg, replace=False)
    idx = sorted(pos_idx.tolist() + neg_idx.tolist())
    sample = rows.loc[idx].reset_index(drop=True)
    logger.info(
        "Pilot sample: %d rows (%d hallucinated, %d grounded), prevalence %.4f, sampling seed %d",
        len(sample), int(sample["label"].sum()), int((sample["label"] == 0).sum()),
        float(sample["label"].mean()), seed,
    )
    return sample


def build_user_prompt(row: pd.Series) -> str:
    return USER_TEMPLATE.format(
        question=str(row["question"]).strip(),
        context=str(row["context"]).strip(),
        answer=str(row["answer"]).strip(),
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _coerce_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(conf):
        return 0.5
    if 2.0 < conf <= 100.0:
        conf /= 100.0
    return float(min(1.0, max(0.0, conf)))


def parse_judge_response(content: str) -> dict[str, Any]:
    """Parse the judge JSON with a conservative keyword fallback."""
    text = (content or "").strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    candidates = [text, fenced]
    data: dict[str, Any] | None = None
    for candidate in candidates:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            data = parsed
            break

    if data is not None:
        raw_judgment = str(data.get("judgment", data.get("judgement", ""))).strip().lower()
        reason = str(data.get("reason", data.get("reasoning", ""))).strip()[:300]
        confidence = _coerce_confidence(data.get("confidence", 0.5))
        hallucinated_tokens = ("hallucinat", "unsupported", "not grounded", "contradict", "unfaithful")
        grounded_tokens = ("grounded", "supported", "faithful", "not hallucinated", "no hallucination")
        if any(tok in raw_judgment for tok in ("not hallucinated", "no hallucination", "not contradicted", "not unfaithful")):
            judgment = "grounded"
        elif any(tok in raw_judgment for tok in hallucinated_tokens):
            judgment = "hallucinated"
        elif any(tok in raw_judgment for tok in grounded_tokens):
            judgment = "grounded"
        else:
            judgment = ""
        if judgment:
            return {
                "judgment": judgment,
                "confidence": confidence,
                "reason": reason,
                "parse_ok": True,
            }

    # Fallback: only used when the model does not return parseable JSON.
    lowered = text.lower()
    if "not hallucinated" in lowered or "no hallucination" in lowered or "not contradicted" in lowered:
        judgment = "grounded"
    elif "hallucinated" in lowered and "grounded" not in lowered:
        judgment = "hallucinated"
    elif "grounded" in lowered or "supported" in lowered:
        judgment = "grounded"
    else:
        judgment = "grounded"
    return {
        "judgment": judgment,
        "confidence": 0.5,
        "reason": text[:300],
        "parse_ok": False,
    }


def hallucination_score(judgment: str, confidence: float) -> float:
    """Convert confidence-in-judgment to a P(hallucinated) score for AUROC."""
    conf = float(min(1.0, max(0.0, confidence)))
    return conf if judgment == "hallucinated" else 1.0 - conf


# ---------------------------------------------------------------------------
# API calls with retries
# ---------------------------------------------------------------------------


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(exc, "code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _retryable(exc: BaseException) -> bool:
    code = _status_code(exc)
    if code in (400, 401, 403, 404):
        return False
    return True


def _retry(fn, *, attempts: int, base_delay: float, label: str):
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise the final error
            last = exc
            if attempt == attempts - 1 or not _retryable(exc):
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0.0, 0.5)
            logger.warning("%s failed (%s: %s); retry %d/%d in %.1fs",
                           label, type(exc).__name__, exc, attempt + 1, attempts - 1, delay)
            time.sleep(delay)
    assert last is not None
    raise last


def _usage_value(usage: Any, key: str, default: Any = 0) -> Any:
    if usage is None:
        return default
    if isinstance(usage, dict):
        return usage.get(key, default)
    return getattr(usage, key, default)


def _nested_usage(usage: Any, parent: str, key: str) -> int:
    node = _usage_value(usage, parent, None)
    if node is None:
        return 0
    return int(_usage_value(node, key, 0) or 0)


def _is_response_format_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return _status_code(exc) == 400 and (
        "response_format" in text or "json_object" in text or "unsupported parameter" in text
    )


def _is_max_tokens_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return _status_code(exc) == 400 and (
        "max_tokens" in text or "max_completion_tokens" in text or "output limit" in text
    )


def openai_call(client, model: str, row: pd.Series, seed: int, max_completion_tokens: int) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(row)},
    ]

    def _once():
        t0 = time.perf_counter()
        cap = int(max_completion_tokens)
        use_response_format = True
        while True:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": cap,
                "seed": seed,
            }
            if use_response_format:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                response = client.chat.completions.create(**kwargs)
                return response, (time.perf_counter() - t0) * 1000.0, cap
            except Exception as exc:  # noqa: BLE001
                # Reasoning models can spend the whole cap before emitting JSON.
                # Retry once per doubling, up to 2,048 tokens, and record the
                # cap that actually produced the response.
                if _is_response_format_error(exc) and use_response_format:
                    use_response_format = False
                    continue
                if _is_max_tokens_error(exc) and cap < 2048:
                    cap = min(2048, cap * 2)
                    continue
                raise

    response, latency_ms, effective_cap = _retry(
        _once, attempts=5, base_delay=1.0, label=f"openai {row['sample_id']} seed={seed}"
    )
    content = (response.choices[0].message.content or "").strip()
    parsed = parse_judge_response(content)
    usage = getattr(response, "usage", None)
    prompt_tokens = int(_usage_value(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(_usage_value(usage, "completion_tokens", 0) or 0)
    return {
        "backend": "openai",
        "model": model,
        "sample_id": str(row["sample_id"]),
        "seed": int(seed),
        "label": int(row["label"]),
        "judgment": parsed["judgment"],
        "confidence": parsed["confidence"],
        "score": hallucination_score(parsed["judgment"], parsed["confidence"]),
        "reason": parsed["reason"],
        "parse_ok": bool(parsed["parse_ok"]),
        "raw": content[:2000],
        "latency_ms": round(latency_ms, 2),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": _nested_usage(usage, "prompt_tokens_details", "cached_tokens"),
        "completion_tokens": completion_tokens,
        "reasoning_tokens": _nested_usage(usage, "completion_tokens_details", "reasoning_tokens"),
        "max_completion_tokens": int(effective_cap),
        "error": None,
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def check_ollama(url: str, model: str) -> list[str]:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/tags", timeout=10) as resp:
            payload = json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Ollama is not reachable at {url}: {exc}. Start `ollama serve` or set HALU_QWEN_URL."
        ) from exc
    names = [str(item.get("name", "")) for item in payload.get("models", [])]
    base_match = any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)
    if not base_match:
        raise SystemExit(
            f"Ollama model {model!r} is not installed. Available: {names}. "
            f"Run `ollama pull {model}` first."
        )
    return names


def qwen_call(url: str, model: str, row: pd.Series, seed: int, max_predict: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "seed": int(seed),
            "num_predict": int(max_predict),
            "num_ctx": 8192,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    def _once():
        t0 = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as resp:
            data = json.loads(resp.read())
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return data, latency_ms

    data, latency_ms = _retry(
        _once, attempts=4, base_delay=2.0, label=f"qwen {row['sample_id']} seed={seed}"
    )
    content = (data.get("message", {}).get("content") or "").strip()
    parsed = parse_judge_response(content)
    prompt_tokens = int(data.get("prompt_eval_count") or 0)
    completion_tokens = int(data.get("eval_count") or 0)
    return {
        "backend": "qwen",
        "model": model,
        "sample_id": str(row["sample_id"]),
        "seed": int(seed),
        "label": int(row["label"]),
        "judgment": parsed["judgment"],
        "confidence": parsed["confidence"],
        "score": hallucination_score(parsed["judgment"], parsed["confidence"]),
        "reason": parsed["reason"],
        "parse_ok": bool(parsed["parse_ok"]),
        "raw": content[:2000],
        "latency_ms": round(latency_ms, 2),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": 0,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": 0,
        "max_predict": int(max_predict),
        "total_duration_ms": round(float(data.get("total_duration") or 0) / 1e6, 2),
        "load_duration_ms": round(float(data.get("load_duration") or 0) / 1e6, 2),
        "prompt_eval_duration_ms": round(float(data.get("prompt_eval_duration") or 0) / 1e6, 2),
        "eval_duration_ms": round(float(data.get("eval_duration") or 0) / 1e6, 2),
        "error": None,
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def _call_one(args: tuple, backend: str, model: str, url: str, max_completion_tokens: int, max_predict: int) -> dict[str, Any]:
    row, seed, client = args
    try:
        if backend == "openai":
            return openai_call(client, model, row, seed, max_completion_tokens)
        return qwen_call(url, model, row, seed, max_predict)
    except Exception as exc:  # noqa: BLE001 - one failed call must not kill the run
        logger.error("call failed for %s seed=%d: %s: %s", row["sample_id"], seed, type(exc).__name__, exc)
        return {
            "backend": backend,
            "model": model,
            "sample_id": str(row["sample_id"]),
            "seed": int(seed),
            "label": int(row["label"]),
            "judgment": None,
            "confidence": None,
            "score": None,
            "reason": "",
            "parse_ok": False,
            "raw": "",
            "latency_ms": None,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping malformed JSONL line in %s", path)
    return records


def run_judge(
    rows: pd.DataFrame,
    *,
    backend: str,
    seeds: list[int],
    model: str,
    concurrency: int,
    out_tag: str,
    max_completion_tokens: int,
    max_predict: int,
    url: str,
    fresh: bool,
) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = RESULTS_DIR / f"calls_{backend}_{out_tag}.jsonl"
    if fresh and jsonl_path.exists():
        backup = jsonl_path.with_suffix(jsonl_path.suffix + ".bak")
        if backup.exists():
            backup.unlink()
        jsonl_path.rename(backup)
        logger.info("Moved existing checkpoint to %s", backup)

    existing = read_jsonl(jsonl_path)
    if existing and existing[0].get("model") != model:
        raise SystemExit(
            f"Checkpoint {jsonl_path} was produced by model {existing[0].get('model')!r}, "
            f"not {model!r}. Re-run with --fresh or a different --out-tag."
        )
    done = {
        (str(rec.get("sample_id")), int(rec.get("seed")))
        for rec in existing
        if rec.get("judgment") is not None and not rec.get("error")
    }

    tasks = [
        (row, int(seed))
        for _, row in rows.iterrows()
        for seed in seeds
        if (str(row["sample_id"]), int(seed)) not in done
    ]
    logger.info("%s: %d planned calls, %d already complete, %d remaining",
                backend, len(rows) * len(seeds), len(done), len(tasks))

    if tasks:
        if backend == "openai":
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise SystemExit("OPENAI_API_KEY is not set; add it to the root .env before running.")
            client = OpenAI(api_key=api_key, timeout=180.0, max_retries=0)
        else:
            available = check_ollama(url, model)
            logger.info("Ollama models available: %s", available)
            client = None

        lock = threading.Lock()
        completed = 0
        failures = 0
        total = len(tasks)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
                futures = [
                    pool.submit(
                        _call_one,
                        (row, seed, client),
                        backend,
                        model,
                        url,
                        max_completion_tokens,
                        max_predict,
                    )
                    for row, seed in tasks
                ]
                for future in as_completed(futures):
                    record = future.result()
                    with lock:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        completed += 1
                        if record.get("error"):
                            failures += 1
                        if completed % 20 == 0 or completed == total:
                            logger.info("%s progress %d/%d (errors %d)", backend, completed, total, failures)
        if failures == total:
            first = read_jsonl(jsonl_path)[-1].get("error", "unknown error")
            raise SystemExit(f"All {total} calls failed. First error: {first}")

    records = read_jsonl(jsonl_path)
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame["seed"] = frame["seed"].astype(int)
    frame["label"] = frame["label"].astype(int)
    predictions = frame[frame["judgment"].notna()].copy()
    predictions["pred"] = (predictions["judgment"] == "hallucinated").astype(int)
    predictions["score"] = predictions["score"].astype(float)
    predictions = predictions.sort_values(["sample_id", "seed"]).reset_index(drop=True)
    predictions.to_csv(RESULTS_DIR / f"predictions_{backend}_{out_tag}.csv", index=False)
    logger.info("Saved %d predictions to %s", len(predictions), f"predictions_{backend}_{out_tag}.csv")
    return frame


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _safe_metric(fn, y: np.ndarray, pred: np.ndarray) -> float | None:
    try:
        return float(fn(y, pred))
    except Exception:  # noqa: BLE001
        return None


def metric_block(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    """Point metrics for one judge repeat."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    score = np.asarray(score, dtype=float)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    auroc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()) if len(y) else None,
        "accuracy": float(accuracy_score(y, pred)) if len(y) else None,
        "precision": float(precision_score(y, pred, zero_division=0)) if len(y) else None,
        "recall": float(recall_score(y, pred, zero_division=0)) if len(y) else None,
        "f1": float(f1_score(y, pred, zero_division=0)) if len(y) else None,
        "auroc": auroc,
        "predicted_positive_rate": float(pred.mean()) if len(pred) else None,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def bootstrap_cis(
    y: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, list[float] | None]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    score = np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    funcs = {
        "accuracy": lambda yy, pp: accuracy_score(yy, pp),
        "precision": lambda yy, pp: precision_score(yy, pp, zero_division=0),
        "recall": lambda yy, pp: recall_score(yy, pp, zero_division=0),
        "f1": lambda yy, pp: f1_score(yy, pp, zero_division=0),
        "predicted_positive_rate": lambda yy, pp: float(np.mean(pp)),
    }
    values: dict[str, list[float]] = {key: [] for key in funcs}
    values["auroc"] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yy, pp, ss = y[idx], pred[idx], score[idx]
        if len(np.unique(yy)) < 2:
            continue
        for key, fn in funcs.items():
            values[key].append(float(fn(yy, pp)))
        values["auroc"].append(float(roc_auc_score(yy, ss)))
    out: dict[str, list[float] | None] = {}
    for key, vals in values.items():
        if len(vals) < 50:
            out[key] = None
        else:
            out[key] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return out


def _mean_sd(values: list[float | None]) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"mean": None, "sd": None}
    return {
        "mean": float(np.mean(clean)),
        "sd": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
    }


def _kappa(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        value = float(cohen_kappa_score(a, b))
        return value if math.isfinite(value) else None
    except Exception:  # noqa: BLE001
        return None


def agreement_block(predictions: pd.DataFrame, seeds: list[int], test_rows: pd.DataFrame) -> dict[str, Any]:
    pivot = predictions.pivot_table(index="sample_id", columns="seed", values="pred", aggfunc="first")
    complete = pivot.dropna(subset=[s for s in seeds if s in pivot.columns])
    out: dict[str, Any] = {"n_common_rows": int(len(complete)), "pairwise": {}}
    for a, b in combinations([s for s in seeds if s in complete.columns], 2):
        x = complete[a].to_numpy(dtype=int)
        y = complete[b].to_numpy(dtype=int)
        out["pairwise"][f"{a}_vs_{b}"] = {
            "n": int(len(x)),
            "exact_match": float(np.mean(x == y)),
            "cohen_kappa": _kappa(x, y),
        }
    pairwise_kappas = [
        value["cohen_kappa"]
        for value in out["pairwise"].values()
        if value["cohen_kappa"] is not None
    ]
    out["mean_pairwise_cohen_kappa"] = float(np.mean(pairwise_kappas)) if pairwise_kappas else None
    if complete.shape[1] == len(seeds) and complete.shape[1] >= 2:
        matrix = complete.to_numpy(dtype=int)
        out["three_way_exact_match"] = float(np.mean(np.all(matrix == matrix[:, [0]], axis=1)))
        counts = np.column_stack([(matrix == 0).sum(axis=1), (matrix == 1).sum(axis=1)])
        try:
            from statsmodels.stats.inter_rater import fleiss_kappa

            value = float(fleiss_kappa(counts, method="fleiss"))
            out["fleiss_kappa"] = value if math.isfinite(value) else None
        except Exception:  # noqa: BLE001
            out["fleiss_kappa"] = None
        majority = (matrix.mean(axis=1) >= 0.5).astype(int)
        labels = complete.index.map(test_rows.set_index("sample_id")["label"].to_dict()).to_numpy(dtype=int)
        out["majority_vote_vs_label"] = {
            "n": int(len(labels)),
            "exact_match": float(np.mean(majority == labels)),
            "cohen_kappa": _kappa(majority, labels),
        }
        out["majority_vote"] = {
            "predicted_positive_rate": float(majority.mean()),
            "n_unanimous": int(np.sum(np.all(matrix == matrix[:, [0]], axis=1))),
        }
    # Agreement with the frozen EC-XGB reference (m3, seed 42) on the same rows.
    try:
        ref = pd.read_parquet(PREDICTIONS_PATH)
        ref = ref[(ref["dataset"] == DATASET) & (ref["variant"] == "m3") & (ref["seed"] == 42)]
        ref = ref.set_index("sample_id")["score"]
        out["vs_ec_xgb_seed42"] = {}
        for seed in seeds:
            sub = predictions[predictions["seed"] == seed].set_index("sample_id")
            common = sub.index.intersection(ref.index)
            if len(common) == 0:
                continue
            judge_pred = sub.loc[common, "pred"].to_numpy(dtype=int)
            ec_pred = (ref.loc[common].to_numpy(dtype=float) >= 0.5).astype(int)
            out["vs_ec_xgb_seed42"][str(seed)] = {
                "n": int(len(common)),
                "exact_match": float(np.mean(judge_pred == ec_pred)),
                "cohen_kappa": _kappa(judge_pred, ec_pred),
            }
    except Exception as exc:  # noqa: BLE001
        out["vs_ec_xgb_seed42"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def token_accounting(records: pd.DataFrame) -> dict[str, Any]:
    ok = records[records["judgment"].notna()]
    if ok.empty:
        return {"calls": 0}
    prompt = int(ok["prompt_tokens"].fillna(0).sum())
    cached = int(ok["cached_tokens"].fillna(0).sum())
    completion = int(ok["completion_tokens"].fillna(0).sum())
    reasoning = int(ok["reasoning_tokens"].fillna(0).sum()) if "reasoning_tokens" in ok else 0
    return {
        "calls": int(len(ok)),
        "prompt_tokens": prompt,
        "cached_prompt_tokens": cached,
        "uncached_prompt_tokens": max(0, prompt - cached),
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "total_tokens": prompt + completion,
        "mean_prompt_tokens": float(prompt / len(ok)),
        "mean_completion_tokens": float(completion / len(ok)),
        "cache_hit_rate": float(cached / prompt) if prompt else None,
    }


def estimate_cost(tokens: dict[str, Any], backend: str = "openai") -> dict[str, Any]:
    if not tokens.get("calls"):
        return {"usd": 0.0, "usd_per_1000_calls": None, "pricing": PRICING}
    if backend == "qwen":
        return {
            "usd": 0.0,
            "usd_per_1000_calls": 0.0,
            "pricing": None,
            "assumption": "Local Ollama inference; no API charge. Electricity and hardware are excluded.",
        }
    usd = (
        tokens["uncached_prompt_tokens"] / 1e6 * PRICING["input_per_mtok"]
        + tokens["cached_prompt_tokens"] / 1e6 * PRICING["cached_input_per_mtok"]
        + tokens["completion_tokens"] / 1e6 * PRICING["output_per_mtok"]
    )
    return {
        "usd": float(usd),
        "usd_per_1000_calls": float(usd / tokens["calls"] * 1000),
        "pricing": PRICING,
    }


def latency_summary(records: pd.DataFrame) -> dict[str, Any]:
    ok = records[records["judgment"].notna()]
    values = pd.to_numeric(ok["latency_ms"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def build_metrics(
    records: pd.DataFrame,
    *,
    backend: str,
    model: str,
    seeds: list[int],
    out_tag: str,
    test_rows: pd.DataFrame,
) -> dict[str, Any]:
    ok = records[records["judgment"].notna()].copy()
    ok["pred"] = (ok["judgment"] == "hallucinated").astype(int)
    ok["score"] = ok["score"].astype(float)

    per_seed: dict[str, Any] = {}
    for seed in seeds:
        sub = ok[ok["seed"] == seed]
        if sub.empty:
            continue
        y = sub["label"].to_numpy(dtype=int)
        pred = sub["pred"].to_numpy(dtype=int)
        score = sub["score"].to_numpy(dtype=float)
        block = metric_block(y, pred, score)
        block["ci95"] = bootstrap_cis(y, pred, score)
        block["parse_failures"] = int((~sub["parse_ok"].astype(bool)).sum())
        per_seed[str(seed)] = block

    mean_across_seeds = {}
    for key in ["accuracy", "precision", "recall", "f1", "auroc", "predicted_positive_rate"]:
        mean_across_seeds[key] = _mean_sd([per_seed[s].get(key) for s in per_seed])

    tokens = token_accounting(records)
    errors = records[records["error"].notna()] if "error" in records else records.iloc[0:0]
    if backend == "openai":
        col = "max_completion_tokens"
        max_output = int(ok[col].dropna().iloc[0]) if col in ok and ok[col].notna().any() else MAX_COMPLETION_TOKENS
    else:
        col = "max_predict"
        max_output = int(ok[col].dropna().iloc[0]) if col in ok and ok[col].notna().any() else QWEN_MAX_PREDICT
    metrics = {
        "meta": {
            "work_package": "WP6",
            "backend": backend,
            "model": model,
            "dataset": DATASET,
            "out_tag": out_tag,
            "n_rows": int(ok["sample_id"].nunique()),
            "n_predictions": int(len(ok)),
            "seeds": seeds,
            "prevalence": float(test_rows["label"].mean()),
            "max_completion_tokens": max_output,
            "system_prompt_sha256": PROMPT_SHA256,
            "system_prompt_tokens_o200k": SYSTEM_PROMPT_TOKENS,
            "prompt_doc": "docs/rebuild/judge_prompt.md",
            "pilot": bool(out_tag != "full" or ok["sample_id"].nunique() < EXPECTED_N),
            "note": "Pilot numbers are not final; only --limit 0 produces the WP6 table.",
        },
        "per_seed": per_seed,
        "mean_across_seeds": mean_across_seeds,
        "agreement": agreement_block(ok, seeds, test_rows),
        "tokens": tokens,
        "cost": estimate_cost(tokens, backend),
        "latency": latency_summary(records),
        "errors": {
            "n_failed_calls": int(len(errors)),
            "first_errors": errors["error"].head(5).tolist() if not errors.empty else [],
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"metrics_{backend}_{out_tag}.json"
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    logger.info("Saved metrics to %s", path)
    return metrics


# ---------------------------------------------------------------------------
# Comparison rows: EC-XGB (m3) and length-only baseline
# ---------------------------------------------------------------------------


def _calibration_matrix(names: list[str]) -> pd.DataFrame:
    cal = pd.read_parquet(CALIBRATION_FEATURES_PATH)
    if len(cal) != 5034:
        raise RuntimeError(f"expected 5,034 calibration rows, found {len(cal)}")
    base_cols = [name for name in names if name != "source_ragtruth"]
    matrix = cal[base_cols].copy()
    matrix["source_ragtruth"] = 1.0
    for col in ["n_chars", "n_words"]:
        matrix[col] = np.log1p(np.clip(matrix[col].astype(float), 0.0, None))
    matrix = matrix[names].astype(float)
    return matrix


def _m3_calibration_scores(seeds: list[int]) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    import joblib

    names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))["m3"]
    cal = pd.read_parquet(CALIBRATION_FEATURES_PATH)
    matrix = _calibration_matrix(names)
    scores: dict[int, np.ndarray] = {}
    for seed in seeds:
        path = FIXED_HP_MODELS / f"xgboost_m3_seed_{seed}.joblib"
        if not path.exists():
            raise RuntimeError(f"missing m3 model for seed {seed}: {path}")
        model = joblib.load(path)
        try:
            model.get_booster().set_param({"device": "cpu"})
        except Exception:  # noqa: BLE001
            pass
        scores[seed] = model.predict_proba(matrix)[:, 1].astype(float)
    return cal, scores


def _threshold_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float = 0.05) -> float:
    negatives = np.asarray(scores, dtype=float)[np.asarray(labels, dtype=int) == 0]
    if len(negatives) == 0:
        raise RuntimeError("cannot set an FPR threshold without negative rows")
    # Pick the smallest observed score whose tail mass is at most target_fpr.
    # This avoids ties pushing the empirical FPR above the requested level.
    unique = np.unique(negatives)
    sorted_neg = np.sort(negatives)
    counts_ge = len(negatives) - np.searchsorted(sorted_neg, unique, side="left")
    fractions = counts_ge / len(negatives)
    valid = unique[fractions <= target_fpr]
    if len(valid):
        return float(valid[0])
    return float(np.nextafter(unique[-1], np.inf))


def _operating_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (np.asarray(score, dtype=float) >= float(threshold)).astype(int)
    block = metric_block(np.asarray(y, dtype=int), pred, np.asarray(score, dtype=float))
    y = np.asarray(y, dtype=int)
    neg = int((y == 0).sum())
    block["fpr"] = float(((pred == 1) & (y == 0)).sum() / neg) if neg else None
    block["threshold"] = float(threshold)
    return block


def _bootstrap_operating_ci(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= float(threshold)).astype(int)
    return bootstrap_cis(y, pred, score)


def _row(
    *,
    row_id: str,
    family: str,
    seed: int | str,
    n: int,
    prevalence: float,
    auroc: float | None,
    f1_at_05: float | None,
    precision_at_05: float | None,
    recall_at_05: float | None,
    ppr_at_05: float | None,
    f1_at_5fpr: float | None,
    precision_at_5fpr: float | None,
    recall_at_5fpr: float | None,
    fpr_at_5fpr: float | None,
    threshold_5fpr: float | None,
    note: str,
    ci: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ci = ci or {}
    def _ci(name: str, bound: int) -> float | None:
        interval = ci.get(name)
        if not interval:
            return None
        return float(interval[bound])

    return {
        "row_id": row_id,
        "family": family,
        "seed": seed,
        "n": int(n),
        "prevalence": float(prevalence),
        "auroc": auroc,
        "auroc_ci_low": _ci("auroc", 0),
        "auroc_ci_high": _ci("auroc", 1),
        "f1_at_0.5": f1_at_05,
        "f1_at_0.5_ci_low": _ci("f1_at_0.5", 0),
        "f1_at_0.5_ci_high": _ci("f1_at_0.5", 1),
        "precision_at_0.5": precision_at_05,
        "recall_at_0.5": recall_at_05,
        "predicted_positive_rate_at_0.5": ppr_at_05,
        "f1_at_5pct_fpr": f1_at_5fpr,
        "f1_at_5pct_fpr_ci_low": _ci("f1_at_5pct_fpr", 0),
        "f1_at_5pct_fpr_ci_high": _ci("f1_at_5pct_fpr", 1),
        "precision_at_5pct_fpr": precision_at_5fpr,
        "recall_at_5pct_fpr": recall_at_5fpr,
        "fpr_at_5pct_fpr": fpr_at_5fpr,
        "threshold_at_5pct_fpr": threshold_5fpr,
        "note": note,
    }


def build_comparison_rows(seeds: list[int] | None = None) -> dict[str, Any]:
    """Build the WP6 comparison table on the 900 held-out QA rows."""
    seeds = seeds or DEFAULT_SEEDS
    test = pd.read_parquet(PREDICTIONS_PATH)
    test = test[(test["dataset"] == DATASET) & (test["variant"] == "m3")].copy()
    if test["sample_id"].nunique() != EXPECTED_N:
        raise RuntimeError(f"expected {EXPECTED_N} test ids in {PREDICTIONS_PATH}")
    cal, cal_scores = _m3_calibration_scores(seeds)

    rows: list[dict[str, Any]] = []
    per_seed_rows: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        sub = test[test["seed"] == seed].sort_values("sample_id").reset_index(drop=True)
        y = sub["label"].to_numpy(dtype=int)
        score = sub["score"].to_numpy(dtype=float)
        threshold_5 = _threshold_at_fpr(cal_scores[seed], cal["label"].to_numpy(dtype=int))
        at05 = _operating_metrics(y, score, 0.5)
        at5 = _operating_metrics(y, score, threshold_5)
        ci05 = _bootstrap_operating_ci(y, score, 0.5)
        ci5 = _bootstrap_operating_ci(y, score, threshold_5)
        record = _row(
            row_id=f"ec_xgb_m3_seed{seed}",
            family="ec_xgb_m3",
            seed=seed,
            n=len(sub),
            prevalence=float(y.mean()),
            auroc=float(roc_auc_score(y, score)),
            f1_at_05=at05["f1"],
            precision_at_05=at05["precision"],
            recall_at_05=at05["recall"],
            ppr_at_05=at05["predicted_positive_rate"],
            f1_at_5fpr=at5["f1"],
            precision_at_5fpr=at5["precision"],
            recall_at_5fpr=at5["recall"],
            fpr_at_5fpr=at5["fpr"],
            threshold_5fpr=threshold_5,
            ci={"f1_at_0.5": ci05.get("f1"), "auroc": ci05.get("auroc"), "f1_at_5pct_fpr": ci5.get("f1")},
            note="threshold_at_5pct_fpr chosen on the 5,034-row QA calibration split; test rows never used for threshold selection",
        )
        rows.append(record)
        per_seed_rows[seed] = record

    # Mean row across seeds.
    mean_record = {
        "row_id": "ec_xgb_m3_mean",
        "family": "ec_xgb_m3",
        "seed": "mean",
        "n": EXPECTED_N,
        "prevalence": float(test["label"].mean()),
        "note": "mean +/- sd across seeds 42/123/456; thresholds chosen per seed on the calibration split",
    }
    for key in [
        "auroc", "f1_at_0.5", "precision_at_0.5", "recall_at_0.5",
        "predicted_positive_rate_at_0.5", "f1_at_5pct_fpr", "precision_at_5pct_fpr",
        "recall_at_5pct_fpr", "fpr_at_5pct_fpr", "threshold_at_5pct_fpr",
    ]:
        stats = _mean_sd([per_seed_rows[s].get(key) for s in seeds])
        mean_record[key] = stats["mean"]
        mean_record[f"{key}_sd"] = stats["sd"]
    rows.append(mean_record)

    # Length-only baseline: fit on the 5,034 calibration rows.
    cal_nwords = cal["n_words"].to_numpy(dtype=float)
    lr = LogisticRegression(max_iter=2000).fit(
        np.log1p(cal_nwords).reshape(-1, 1), cal["label"].to_numpy(dtype=int)
    )
    cal_len_scores = lr.predict_proba(np.log1p(cal_nwords).reshape(-1, 1))[:, 1]
    threshold_len = _threshold_at_fpr(cal_len_scores, cal["label"].to_numpy(dtype=int))

    ext = pd.read_parquet(EXTERNAL_FEATURES_PATH, columns=["sample_id", "n_words"]).drop_duplicates("sample_id")
    test_one_seed = test[test["seed"] == seeds[0]].sort_values("sample_id").reset_index(drop=True)
    merged = test_one_seed.merge(ext, on="sample_id", how="left", validate="one_to_one")
    if merged["n_words"].isna().any():
        bad = merged.loc[merged["n_words"].isna(), "sample_id"].head(5).tolist()
        raise RuntimeError(f"missing n_words for test rows: {bad}")
    y = merged["label"].to_numpy(dtype=int)
    len_scores = lr.predict_proba(np.log1p(merged["n_words"].to_numpy(dtype=float)).reshape(-1, 1))[:, 1]
    len_at05 = _operating_metrics(y, len_scores, 0.5)
    len_at5 = _operating_metrics(y, len_scores, threshold_len)
    len_ci05 = _bootstrap_operating_ci(y, len_scores, 0.5)
    len_ci5 = _bootstrap_operating_ci(y, len_scores, threshold_len)
    rows.append(_row(
        row_id="length_only",
        family="length_only",
        seed="fit_once",
        n=len(merged),
        prevalence=float(y.mean()),
        auroc=float(roc_auc_score(y, len_scores)),
        f1_at_05=len_at05["f1"],
        precision_at_05=len_at05["precision"],
        recall_at_05=len_at05["recall"],
        ppr_at_05=len_at05["predicted_positive_rate"],
        f1_at_5fpr=len_at5["f1"],
        precision_at_5fpr=len_at5["precision"],
        recall_at_5fpr=len_at5["recall"],
        fpr_at_5fpr=len_at5["fpr"],
        threshold_5fpr=threshold_len,
        ci={"f1_at_0.5": len_ci05.get("f1"), "auroc": len_ci05.get("auroc"), "f1_at_5pct_fpr": len_ci5.get("f1")},
        note="logistic regression on log1p(n_words), fit on the 5,034 QA calibration rows; threshold_at_5pct_fpr chosen on the same split",
    ))

    # Reference row: flag everything (the majority-class-free F1 floor at this prevalence).
    prevalence = float(y.mean())
    rows.append({
        "row_id": "all_positive",
        "family": "all_positive",
        "seed": "none",
        "n": int(len(y)),
        "prevalence": prevalence,
        "auroc": None,
        "f1_at_0.5": float(2 * prevalence / (1 + prevalence)),
        "precision_at_0.5": prevalence,
        "recall_at_0.5": 1.0,
        "predicted_positive_rate_at_0.5": 1.0,
        "f1_at_5pct_fpr": None,
        "precision_at_5pct_fpr": None,
        "recall_at_5pct_fpr": None,
        "fpr_at_5pct_fpr": None,
        "threshold_at_5pct_fpr": None,
        "note": "flag-everything reference at the natural prevalence; AUROC undefined",
    })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "comparison_rows.csv", index=False)
    meta = {
        "schema": "wp6-comparison-v1",
        "dataset": DATASET,
        "n_test": EXPECTED_N,
        "seeds": seeds,
        "calibration_rows": 5034,
        "calibration_prevalence": float(cal["label"].mean()),
        "test_prevalence": float(test["label"].mean()),
        "threshold_rule": "smallest observed calibration negative score whose empirical negative tail is at most 5%; applied unchanged to test",
        "auroc_note": "AUROC is threshold-free; it is reported once per model and seed, alongside F1 at 0.5 and at the 5% FPR operating point.",
        "operating_points": {"fixed_threshold": 0.5, "calibrated_threshold": "5% FPR on the 5,034-row QA calibration split"},
        "length_baseline": {
            "features": ["log1p(n_words)"],
            "fit_rows": 5034,
            "coefficient": float(lr.coef_[0, 0]),
            "intercept": float(lr.intercept_[0]),
            "calibration_auroc": float(roc_auc_score(cal["label"], cal_len_scores)),
        },
        "m3_calibration_auroc": {str(s): float(roc_auc_score(cal["label"], cal_scores[s])) for s in seeds},
        "m3_calibration_fpr_at_threshold": {
            str(s): float(
                (cal_scores[s][cal["label"].to_numpy(dtype=int) == 0]
                 >= per_seed_rows[s]["threshold_at_5pct_fpr"]).mean()
            )
            for s in seeds
        },
        "length_calibration_fpr_at_threshold": float(
            (cal_len_scores[cal["label"].to_numpy(dtype=int) == 0] >= threshold_len).mean()
        ),
    }
    (RESULTS_DIR / "comparison_rows.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Saved comparison rows to %s", RESULTS_DIR / "comparison_rows.csv")
    print(frame[[
        "row_id", "n", "auroc", "f1_at_0.5", "predicted_positive_rate_at_0.5",
        "f1_at_5pct_fpr", "fpr_at_5pct_fpr",
    ]].to_string(index=False))
    return {"meta": meta, "rows": rows}


# ---------------------------------------------------------------------------
# Encoder timing
# ---------------------------------------------------------------------------


def time_encoder(
    rows: pd.DataFrame,
    *,
    model_id: str = "roberta-base",
    batch_size: int = 8,
    max_length: int = 512,
    budget_s: float = 600.0,
) -> dict[str, Any]:
    """Time RoBERTa-base inference on up to 200 rows (batch 8, fp16 on CUDA)."""
    out: dict[str, Any] = {
        "schema": "wp6-encoder-timing-v1",
        "model_id": model_id,
        "requested_batch_size": batch_size,
        "max_length": max_length,
        "n_rows_requested": int(len(rows)),
        "budget_s": budget_s,
        "status": "started",
    }
    try:
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        out.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        _write_encoder_timing(out)
        return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    out.update({
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    })

    started = time.perf_counter()
    try:
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, num_labels=2, dtype=dtype
            )
        except TypeError:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, num_labels=2, torch_dtype=dtype
            )
        model.to(device).eval()
        out["model_load_s"] = round(time.perf_counter() - load_start, 3)
    except Exception as exc:  # noqa: BLE001
        out.update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.perf_counter() - started, 3),
        })
        _write_encoder_timing(out)
        return out

    texts = [
        f"{str(row['question']).strip()}\n{str(row['context']).strip()}\n{str(row['answer']).strip()}"
        for _, row in rows.iterrows()
    ]
    tokenize_start = time.perf_counter()
    features = [
        tokenizer(text, truncation=True, max_length=max_length, return_attention_mask=True)
        for text in texts
    ]
    out["tokenize_s"] = round(time.perf_counter() - tokenize_start, 3)
    out["n_rows_tokenized"] = int(len(features))
    out["n_rows_truncated"] = int(sum(len(f["input_ids"]) >= max_length for f in features))

    def _run(batch_features):
        batch = tokenizer.pad(batch_features, padding=True, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model(**batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    # Warmup on the first batch, excluded from the timed result.
    warmup_ms = None
    if features:
        try:
            warmup_ms = _run(features[:batch_size])
        except Exception as exc:  # noqa: BLE001
            out.update({
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.perf_counter() - started, 3),
            })
            _write_encoder_timing(out)
            return out
    out["warmup_batch_ms"] = round(warmup_ms, 2) if warmup_ms is not None else None

    batch_ms: list[float] = []
    timed_rows = 0
    stopped_early = False
    for start in range(0, len(features), batch_size):
        if time.perf_counter() - started > budget_s:
            stopped_early = True
            break
        chunk = features[start : start + batch_size]
        try:
            batch_ms.append(_run(chunk))
        except Exception as exc:  # noqa: BLE001
            out.update({
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.perf_counter() - started, 3),
            })
            _write_encoder_timing(out)
            return out
        timed_rows += len(chunk)

    total_ms = float(np.sum(batch_ms)) if batch_ms else 0.0
    out.update({
        "status": "partial" if stopped_early else "ok",
        "n_rows_timed": int(timed_rows),
        "n_batches_timed": int(len(batch_ms)),
        "timed_total_s": round(total_ms / 1000.0, 3),
        "ms_per_row": round(total_ms / timed_rows, 2) if timed_rows else None,
        "ms_per_batch_mean": round(float(np.mean(batch_ms)), 2) if batch_ms else None,
        "ms_per_batch_p50": round(float(np.median(batch_ms)), 2) if batch_ms else None,
        "ms_per_batch_p95": round(float(np.percentile(batch_ms, 95)), 2) if batch_ms else None,
        "rows_per_s": round(timed_rows / (total_ms / 1000.0), 2) if total_ms else None,
        "stopped_early": stopped_early,
        "elapsed_s": round(time.perf_counter() - started, 3),
    })
    if device.type == "cuda":
        out["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    _write_encoder_timing(out)
    return out


def _write_encoder_timing(out: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "encoder_timing.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Saved encoder timing to %s", RESULTS_DIR / "encoder_timing.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WP6 judge and encoder comparison")
    parser.add_argument("--backend", choices=["openai", "qwen"], default="openai")
    parser.add_argument("--limit", type=int, default=DEFAULT_PILOT_N,
                        help="number of test rows; 0 = all 900 (default: 30-row pilot)")
    parser.add_argument("--repeats", type=int, default=3, help="number of seeds (default: 3)")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds (overrides --repeats)")
    parser.add_argument("--concurrency", type=int, default=0, help="0 = auto (6 OpenAI, 1 Qwen)")
    parser.add_argument("--out-tag", default=None, help="artifact suffix (default: pilot or full)")
    parser.add_argument("--pilot-seed", type=int, default=42, help="sampling seed for a pilot subset")
    parser.add_argument("--model", default=None, help="model name; defaults to OPENAI_MODEL or QWEN_MODEL")
    parser.add_argument("--max-completion-tokens", type=int, default=MAX_COMPLETION_TOKENS)
    parser.add_argument("--max-predict", type=int, default=QWEN_MAX_PREDICT)
    parser.add_argument("--qwen-url", default=QWEN_URL)
    parser.add_argument("--comparisons-only", action="store_true", help="build the m3 and length-only table and exit")
    parser.add_argument("--metrics-only", action="store_true", help="recompute metrics from an existing checkpoint and exit")
    parser.add_argument("--time-encoder", action="store_true", help="time RoBERTa-base on 200 rows and exit")
    parser.add_argument("--encoder-rows", type=int, default=200)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--encoder-model", default="roberta-base")
    parser.add_argument("--fresh", action="store_true", help="ignore an existing JSONL checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="print the projected call count and cost and exit")
    return parser.parse_args(argv)


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
        if not seeds:
            raise SystemExit("--seeds must contain at least one integer")
        return seeds
    if args.repeats < 1 or args.repeats > len(DEFAULT_SEEDS):
        raise SystemExit(f"--repeats must be between 1 and {len(DEFAULT_SEEDS)}")
    return DEFAULT_SEEDS[: args.repeats]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_test_rows()

    if args.time_encoder:
        n_encoder = max(1, min(args.encoder_rows, len(rows)))
        sample = rows.sample(n=n_encoder, random_state=args.pilot_seed).sort_index().reset_index(drop=True)
        timing = time_encoder(
            sample,
            model_id=args.encoder_model,
            batch_size=args.encoder_batch_size,
        )
        print(json.dumps(timing, indent=2, default=str))
        return

    if args.comparisons_only:
        result = build_comparison_rows()
        print(json.dumps(result["meta"], indent=2, default=str))
        return

    seeds = resolve_seeds(args)
    model = args.model or (
        os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
        if args.backend == "openai"
        else os.environ.get("QWEN_MODEL", QWEN_MODEL)
    )
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    if args.limit == 0 or args.limit >= len(rows):
        sample = rows.reset_index(drop=True)
        out_tag = args.out_tag or "full"
    else:
        sample = select_pilot(rows, args.limit, args.pilot_seed)
        out_tag = args.out_tag or "pilot"
    concurrency = args.concurrency or (6 if args.backend == "openai" else 1)

    logger.info("Backend %s | model %s | rows %d | seeds %s | concurrency %d | tag %s",
                args.backend, model, len(sample), seeds, concurrency, out_tag)
    logger.info("System prompt: %d tokens (o200k), sha256 %s", SYSTEM_PROMPT_TOKENS, PROMPT_SHA256[:16])

    if args.dry_run:
        n_calls = len(sample) * len(seeds)
        avg_user_tokens = int(np.mean([count_tokens(build_user_prompt(row)) for _, row in sample.iterrows()]))
        est_prompt = n_calls * (SYSTEM_PROMPT_TOKENS + avg_user_tokens)
        est_completion = n_calls * 120
        if args.backend == "openai":
            est_cost = (
                est_prompt * 0.3 / 1e6 * PRICING["input_per_mtok"]
                + est_prompt * 0.7 / 1e6 * PRICING["cached_input_per_mtok"]
                + est_completion / 1e6 * PRICING["output_per_mtok"]
            )
        else:
            est_cost = 0.0
        print(json.dumps({
            "rows": len(sample),
            "seeds": seeds,
            "calls": n_calls,
            "estimated_prompt_tokens": est_prompt,
            "estimated_completion_tokens": est_completion,
            "estimated_cost_usd": round(float(est_cost), 4),
            "pricing": PRICING,
            "note": "Estimate assumes 70% of the static prefix is cached and 120 output tokens per call.",
        }, indent=2))
        return

    if args.metrics_only:
        path = RESULTS_DIR / f"calls_{args.backend}_{out_tag}.jsonl"
        records = pd.DataFrame(read_jsonl(path))
        if records.empty:
            raise SystemExit(f"no checkpoint records at {path}")
        metrics = build_metrics(
            records, backend=args.backend, model=model, seeds=seeds,
            out_tag=out_tag, test_rows=rows,
        )
        print(json.dumps(metrics["mean_across_seeds"], indent=2, default=str))
        return

    records = run_judge(
        sample,
        backend=args.backend,
        seeds=seeds,
        model=model,
        concurrency=concurrency,
        out_tag=out_tag,
        max_completion_tokens=args.max_completion_tokens,
        max_predict=args.max_predict,
        url=args.qwen_url,
        fresh=args.fresh,
    )
    if records.empty:
        raise SystemExit("No records were produced.")
    metrics = build_metrics(
        records, backend=args.backend, model=model, seeds=seeds,
        out_tag=out_tag, test_rows=rows,
    )
    print(json.dumps({
        "meta": metrics["meta"],
        "mean_across_seeds": metrics["mean_across_seeds"],
        "agreement": metrics["agreement"],
        "tokens": metrics["tokens"],
        "cost": metrics["cost"],
        "latency": metrics["latency"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
