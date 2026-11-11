"""P0.3 / P0.4 judge experiments on HaluEval (branch ``rebuild``).

Two experiments built on the WP6 judge implementation in
``src/models/rebuild_judge.py`` (same static prompt, request settings, parser,
and metric helpers):

  * ``length-matched``: GPT-5.6-Luna (seeds 42/123/456) and Qwen 3.5 4B
    (seed 42) on the 158-row exact-length-matched HaluEval subset, compared
    with fixed-hp EC-XGB (m3) and the length-only logistic baseline on the
    same rows.
  * ``prompt-sensitivity``: 100 HaluEval test rows, paired and stratified by
    label (50/50), three system prompts (current, strict evidence-only,
    concise JSON), GPT 3 seeds, Qwen 1 seed.

Every call is checkpointed to a JSONL file immediately.  The OpenAI key is
loaded from the repo-root ``.env``; it is never printed or written to an
artifact.

Usage (repo root, ``.venv``)::

    python src/models/rebuild_p0_judge.py length-matched
    python src/models/rebuild_p0_judge.py prompt-sensitivity
    python src/models/rebuild_p0_judge.py all
    python src/models/rebuild_p0_judge.py report

Writes under ``artifacts/results/rebuild/p0/``.  Prompt text is published to
``docs/rebuild/p0/prompt_variants.md`` by ``write_prompt_doc()``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Load the root .env explicitly, as required by the P0 protocol.  The key is
# never printed or written to an artifact.
from dotenv import load_dotenv

load_dotenv(".env")
load_dotenv(ROOT / ".env")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.models import rebuild_judge as RJ
from src.models.config import BOOTSTRAP_SEED, N_BOOTSTRAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rebuild_p0_judge")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

OUT_DIR = ROOT / "artifacts" / "results" / "rebuild" / "p0"
REPORT_DIR = OUT_DIR / "judge"
CKPT_DIR = OUT_DIR / "checkpoints"
PROMPT_DOC = ROOT / "docs" / "rebuild" / "p0" / "prompt_variants.md"

QA_PATH = ROOT / "data" / "processed" / "qa_clean.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "features_full.parquet"
MATCHED_IDS_PATH = ROOT / "artifacts" / "results" / "rebuild" / "length_control_fixed_hp_matched_ids.csv"
FIXED_HP_PREDICTIONS = ROOT / "artifacts" / "results" / "rebuild" / "fixed_hp" / "fixed_hp_predictions.parquet"

GPT_SEEDS = [42, 123, 456]
QWEN_SEEDS = [42]
GPT_CONCURRENCY = 6
QWEN_CONCURRENCY = 1
DEFAULT_BUDGET_USD = 1.00

# The current prompt is imported verbatim from rebuild_judge.py.  The two
# additional prompts are fixed before any evaluation; no prompt is selected
# on test performance.
STRICT_EVIDENCE_PROMPT = """You are a strict factual-consistency judge for a research evaluation of hallucination detection. In each request you receive one QUESTION, one CONTEXT, and one ANSWER. Decide whether every factual claim in the ANSWER is explicitly supported by the CONTEXT.

EVIDENCE STANDARD
- A factual claim is any assertion about an entity, event, number, date, quantity, location, cause, or relationship.
- Explicit support means a span in the CONTEXT states the claim directly, or restates it with only surface-level paraphrase. The wording does not need to match, but the same fact must be stated.
- Do not use world knowledge, plausibility, or expectations about what an answer should say.
- Do not infer a fact by combining separate passages unless the CONTEXT itself states the relationship.
- Do not accept a claim because the QUESTION implies it. The QUESTION is not evidence.
- If the ANSWER adds a detail that no CONTEXT span states, the claim is unsupported, even when the detail is true in the real world.
- If a number, date, name, location, or quantity differs from the CONTEXT, the claim is contradicted.

DECISION
1. Identify the checkable factual claims in the ANSWER internally.
2. For each claim, locate the exact CONTEXT span that states it.
3. If every claim has an explicit supporting span, output grounded.
4. If any claim lacks an explicit supporting span or is contradicted, output hallucinated.
5. If the ANSWER has no checkable factual claim, output grounded.
6. If the ANSWER is a refusal or asks for clarification, output grounded.

OUTPUT
Return one JSON object and nothing else. Do not use markdown fences or commentary. The object must have exactly these keys:
{"judgment": "grounded" or "hallucinated", "confidence": <number between 0 and 1>, "reason": "<at most 25 words quoting or naming the deciding CONTEXT span>"}
The reason must identify the deciding span. For grounded, name the main supporting span. For hallucinated, name the missing or contradicted detail.
"""

CONCISE_JSON_PROMPT = """You are a factual-consistency judge. In each request you receive one QUESTION, one CONTEXT, and one ANSWER. Decide whether the ANSWER is grounded in the CONTEXT.

grounded: every factual claim in the ANSWER is supported by the CONTEXT.
hallucinated: at least one factual claim in the ANSWER is contradicted by the CONTEXT or absent from it.

Use only the CONTEXT. Do not use world knowledge. The QUESTION is not evidence. Do not reward length or detail.

Return only this JSON object, with no markdown:
{"judgment": "grounded" or "hallucinated", "confidence": <number between 0 and 1>, "reason": "<at most 25 words naming the deciding claim>"}
"""

PROMPT_VARIANTS: dict[str, dict[str, Any]] = {
    "current": {
        "title": "Current WP6 prompt",
        "system": RJ.SYSTEM_PROMPT,
        "note": "Imported verbatim from SYSTEM_PROMPT in src/models/rebuild_judge.py.",
    },
    "strict_evidence_only": {
        "title": "Strict evidence-only",
        "system": STRICT_EVIDENCE_PROMPT,
        "note": (
            "Requires an explicit CONTEXT span for every factual claim; no inference beyond "
            "surface paraphrase and no cross-passage composition."
        ),
    },
    "concise_json": {
        "title": "Concise JSON",
        "system": CONCISE_JSON_PROMPT,
        "note": "Minimal instruction set with the same output schema.",
    },
}

for _name, _meta in PROMPT_VARIANTS.items():
    _meta["sha256"] = hashlib.sha256(_meta["system"].encode("utf-8")).hexdigest()
    _meta["tokens_o200k"] = RJ.count_tokens(_meta["system"])

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", path)


def _round(value: Any, ndigits: int = 6) -> Any:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(value):
        return None
    return round(value, ndigits)


def _metric_range(values: list[float | None]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"n": 0, "min": None, "max": None, "range": None, "mean": None, "sd": None}
    return {
        "n": len(clean),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "range": float(np.max(clean) - np.min(clean)),
        "mean": float(np.mean(clean)),
        "sd": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_qa() -> pd.DataFrame:
    qa = pd.read_parquet(QA_PATH)
    required = {"sample_id", "item_idx", "question", "context", "answer", "label", "split"}
    missing = required - set(qa.columns)
    if missing:
        raise RuntimeError(f"qa_clean.parquet is missing columns: {sorted(missing)}")
    if qa["sample_id"].duplicated().any():
        raise RuntimeError("qa_clean.parquet has duplicate sample_id values")
    for col in ["question", "context", "answer"]:
        bad = qa[col].isna() | (qa[col].astype(str).str.strip() == "")
        if bad.any():
            raise RuntimeError(f"qa_clean.parquet has {int(bad.sum())} empty {col} values")
    qa["label"] = qa["label"].astype(int)
    return qa


def load_length_matched() -> pd.DataFrame:
    ids = pd.read_csv(MATCHED_IDS_PATH, dtype={"sample_id": str})
    expected = {"sample_id", "bin", "n_words", "label"}
    if not expected.issubset(ids.columns):
        raise RuntimeError(f"{MATCHED_IDS_PATH} is missing columns: {sorted(expected - set(ids.columns))}")
    if len(ids) != 158:
        raise RuntimeError(f"expected 158 matched ids, found {len(ids)}")
    ids = ids.rename(columns={"label": "label_ids"})
    qa = load_qa()
    merged = ids.merge(qa, on="sample_id", how="left", validate="one_to_one")
    if merged["question"].isna().any():
        bad = merged.loc[merged["question"].isna(), "sample_id"].head(5).tolist()
        raise RuntimeError(f"matched ids missing from qa_clean.parquet: {bad}")
    if (merged["label_ids"].astype(int) != merged["label"].astype(int)).any():
        raise RuntimeError("matched-id labels disagree with qa_clean.parquet")
    n_pos = int(merged["label"].sum())
    n_neg = int((merged["label"] == 0).sum())
    if n_pos != 79 or n_neg != 79:
        raise RuntimeError(f"expected 79/79 matched rows, found {n_pos}/{n_neg}")
    merged = merged.sort_values("sample_id").reset_index(drop=True)
    logger.info("Loaded %d length-matched rows (%d hallucinated, %d grounded)", len(merged), n_pos, n_neg)
    return merged


def select_prompt_rows(seed: int = 42, n_items: int = 50) -> pd.DataFrame:
    """Paired, label-stratified HaluEval test sample.

    Each selected ``item_idx`` contributes its correct and hallucinated answer,
    so the sample has 50 grounded and 50 hallucinated rows on 50 distinct
    questions and contexts.  This is stratified by label and controls the
    context across the two labels.
    """
    qa = load_qa()
    test = qa[qa["split"] == "test"].copy()
    pairs = test.groupby("item_idx")["label"].agg(["count", "sum"])
    complete = pairs[(pairs["count"] == 2) & (pairs["sum"] == 1)].index.to_numpy()
    if len(complete) < n_items:
        raise RuntimeError(f"only {len(complete)} complete HaluEval test pairs available")
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(complete, size=n_items, replace=False))
    rows = test[test["item_idx"].isin(chosen)].copy()
    rows = rows.sort_values(["item_idx", "label"]).reset_index(drop=True)
    counts = rows["label"].value_counts().to_dict()
    if counts.get(0) != n_items or counts.get(1) != n_items:
        raise RuntimeError(f"paired sample is not balanced: {counts}")
    logger.info(
        "Prompt-sensitivity sample: %d rows, %d items, labels %s, seed %d",
        len(rows), rows["item_idx"].nunique(), counts, seed,
    )
    return rows


# ---------------------------------------------------------------------------
# Checkpointed judge runs
# ---------------------------------------------------------------------------


def _cost_of_record(record: dict[str, Any], backend: str) -> float:
    if backend != "openai":
        return 0.0
    prompt = int(record.get("prompt_tokens") or 0)
    cached = int(record.get("cached_tokens") or 0)
    completion = int(record.get("completion_tokens") or 0)
    return (
        max(0, prompt - cached) / 1e6 * RJ.PRICING["input_per_mtok"]
        + cached / 1e6 * RJ.PRICING["cached_input_per_mtok"]
        + completion / 1e6 * RJ.PRICING["output_per_mtok"]
    )


def existing_p0_spend() -> float:
    """Conservative sum over every P0 OpenAI checkpoint (retries included)."""
    total = 0.0
    if not CKPT_DIR.exists():
        return 0.0
    for path in CKPT_DIR.glob("*.jsonl"):
        for record in RJ.read_jsonl(path):
            backend = str(record.get("backend") or ("qwen" if "_qwen_" in path.name else "openai"))
            total += _cost_of_record(record, backend)
    return total


def _call_task(
    task: tuple[pd.Series, int, str],
    *,
    backend: str,
    model: str,
    url: str,
    client: Any,
    max_completion_tokens: int,
    max_predict: int,
) -> dict[str, Any]:
    row, seed, prompt_name = task
    meta = PROMPT_VARIANTS[prompt_name]
    try:
        if backend == "openai":
            record = RJ.openai_call(
                client,
                model,
                row,
                int(seed),
                int(max_completion_tokens),
                system_prompt=meta["system"],
            )
        else:
            record = RJ.qwen_call(
                url,
                model,
                row,
                int(seed),
                int(max_predict),
                system_prompt=meta["system"],
            )
    except Exception as exc:  # noqa: BLE001 - one failed call must not kill the run
        logger.error(
            "call failed for %s seed=%d prompt=%s: %s: %s",
            row["sample_id"], seed, prompt_name, type(exc).__name__, exc,
        )
        record = {
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
            "timestamp_utc": _now(),
        }
    record["prompt_variant"] = prompt_name
    record["system_prompt_sha256"] = meta["sha256"]
    record["system_prompt_tokens_o200k"] = int(meta["tokens_o200k"])
    record["experiment"] = "p0"
    return record


def run_calls(
    rows: pd.DataFrame,
    *,
    experiment: str,
    backend: str,
    seeds: list[int],
    prompt_name: str,
    concurrency: int,
    model: str,
    url: str,
    max_completion_tokens: int,
    max_predict: int,
    fresh: bool = False,
    budget_usd: float | None = None,
    spent_usd: float = 0.0,
) -> pd.DataFrame:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = CKPT_DIR / f"calls_{experiment}_{backend}_{prompt_name}.jsonl"
    if fresh and jsonl_path.exists():
        backup = jsonl_path.with_suffix(jsonl_path.suffix + f".bak.{int(time.time())}")
        jsonl_path.rename(backup)
        logger.info("Moved existing checkpoint to %s", backup)

    existing = RJ.read_jsonl(jsonl_path)
    if existing:
        models = {str(rec.get("model")) for rec in existing}
        if model not in models:
            raise SystemExit(
                f"checkpoint {jsonl_path} was produced by {sorted(models)}, not {model!r}. "
                "Use --fresh or a different experiment tag."
            )
    done = {
        (str(rec.get("sample_id")), int(rec.get("seed")))
        for rec in existing
        if rec.get("judgment") is not None and not rec.get("error") and rec.get("prompt_variant") == prompt_name
    }
    tasks = [
        (row, int(seed), prompt_name)
        for _, row in rows.iterrows()
        for seed in seeds
        if (str(row["sample_id"]), int(seed)) not in done
    ]
    logger.info(
        "%s/%s/%s: %d planned, %d complete, %d remaining (concurrency %d)",
        experiment, backend, prompt_name, len(rows) * len(seeds), len(done), len(tasks), concurrency,
    )

    if tasks:
        client = None
        if backend == "openai":
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise SystemExit(
                    "OPENAI_API_KEY is not set. Add it to the repo-root .env, then rerun: "
                    f"python src/models/rebuild_p0_judge.py {experiment}"
                )
            client = OpenAI(api_key=api_key, timeout=180.0, max_retries=0)
        else:
            available = RJ.check_ollama(url, model)
            logger.info("Ollama models available: %s", available)

        lock = threading.Lock()
        stop = threading.Event()
        completed = 0
        failures = 0
        run_cost = 0.0
        total = len(tasks)

        def submit(pool, task):
            return pool.submit(
                _call_task,
                task,
                backend=backend,
                model=model,
                url=url,
                client=client,
                max_completion_tokens=max_completion_tokens,
                max_predict=max_predict,
            )

        with jsonl_path.open("a", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
                iterator = iter(tasks)
                inflight: dict[Any, tuple] = {}
                for _ in range(max(1, concurrency) * 2):
                    try:
                        task = next(iterator)
                    except StopIteration:
                        break
                    inflight[submit(pool, task)] = task

                while inflight:
                    future = next(as_completed(list(inflight)))
                    inflight.pop(future)
                    record = future.result()
                    with lock:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        completed += 1
                        if record.get("error"):
                            failures += 1
                        run_cost += _cost_of_record(record, backend)
                        if completed % 20 == 0 or completed == total:
                            logger.info(
                                "%s/%s/%s progress %d/%d (errors %d, cost $%.4f)",
                                experiment, backend, prompt_name, completed, total, failures, run_cost,
                            )
                        if budget_usd is not None and backend == "openai":
                            if spent_usd + run_cost > budget_usd:
                                stop.set()
                                logger.error(
                                    "Budget cap reached: spent $%.4f + run $%.4f > $%.2f; "
                                    "stopping new submissions.",
                                    spent_usd, run_cost, budget_usd,
                                )
                    if not stop.is_set():
                        try:
                            task = next(iterator)
                        except StopIteration:
                            continue
                        inflight[submit(pool, task)] = task

        if failures == total:
            first = next(
                (rec.get("error") for rec in reversed(RJ.read_jsonl(jsonl_path)) if rec.get("error")),
                "unknown error",
            )
            raise SystemExit(f"All {total} calls failed. First error: {first}")

    records = pd.DataFrame(RJ.read_jsonl(jsonl_path))
    if records.empty:
        return records
    records["seed"] = records["seed"].astype(int)
    records["label"] = records["label"].astype(int)
    logger.info("%s/%s/%s: checkpoint has %d records", experiment, backend, prompt_name, len(records))
    return records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def group_bootstrap_cis(
    y: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, list[float] | None]:
    """Item-level bootstrap CIs (rows inside one item stay together)."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    score = np.asarray(score, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in unique}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "accuracy": [], "precision": [], "recall": [], "f1": [], "auroc": [], "flagged_rate": [],
    }
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index[g] for g in chosen])
        yy, pp, ss = y[idx], pred[idx], score[idx]
        if len(np.unique(yy)) < 2:
            continue
        values["accuracy"].append(float((yy == pp).mean()))
        tp = int(((pp == 1) & (yy == 1)).sum())
        fp = int(((pp == 1) & (yy == 0)).sum())
        fn = int(((pp == 0) & (yy == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values["precision"].append(float(precision))
        values["recall"].append(float(recall))
        values["f1"].append(float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
        values["auroc"].append(float(roc_auc_score(yy, ss)))
        values["flagged_rate"].append(float(pp.mean()))
    out: dict[str, list[float] | None] = {}
    for key, vals in values.items():
        if len(vals) < 50:
            out[key] = None
        else:
            out[key] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return out


def _ci(block: dict[str, Any], key: str, bound: int) -> float | None:
    interval = block.get("ci95", {}).get(key)
    if not interval:
        return None
    return float(interval[bound])


def judge_metric_block(
    y: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    block = RJ.metric_block(y, pred, score)
    block["flagged_rate"] = block.pop("predicted_positive_rate")
    block["ci95"] = group_bootstrap_cis(y, pred, score, groups)
    return block


def baseline_metric_block(y: np.ndarray, score: np.ndarray, groups: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = (np.asarray(score, dtype=float) >= float(threshold)).astype(int)
    block = RJ.metric_block(y, pred, score)
    block["flagged_rate"] = block.pop("predicted_positive_rate")
    block["threshold"] = float(threshold)
    block["ci95"] = group_bootstrap_cis(y, pred, score, groups)
    return block


def _pairwise(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return {"n": 0, "exact_match": None, "cohen_kappa": None}
    x = a.loc[common].to_numpy(dtype=int)
    y = b.loc[common].to_numpy(dtype=int)
    return {
        "n": int(len(common)),
        "exact_match": float(np.mean(x == y)),
        "cohen_kappa": RJ._kappa(x, y),
    }


def _majority_vote(pivot: pd.DataFrame) -> pd.Series:
    return (pivot.mean(axis=1) >= 0.5).astype(int)


def _labels_for(index: pd.Index, rows: pd.DataFrame) -> np.ndarray:
    mapping = rows.set_index("sample_id")["label"].to_dict()
    return index.map(mapping).to_numpy(dtype=int)


def build_judge_metrics(
    records: pd.DataFrame,
    *,
    rows: pd.DataFrame,
    seeds: list[int],
    backend: str,
    model: str,
) -> dict[str, Any]:
    ok = records[records["judgment"].notna()].copy()
    ok["pred"] = (ok["judgment"] == "hallucinated").astype(int)
    ok["score"] = ok["score"].astype(float)
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        sub = ok[ok["seed"] == seed].sort_values("sample_id")
        if sub.empty:
            continue
        merged = sub.merge(rows[["sample_id", "item_idx"]], on="sample_id", how="left", validate="many_to_one")
        block = judge_metric_block(
            merged["label"].to_numpy(dtype=int),
            merged["pred"].to_numpy(dtype=int),
            merged["score"].to_numpy(dtype=float),
            merged["item_idx"].to_numpy(),
        )
        block["parse_failures"] = int((~merged["parse_ok"].astype(bool)).sum())
        per_seed[str(seed)] = block

    mean_across_seeds = {
        key: RJ._mean_sd([per_seed[s].get(key) for s in per_seed])
        for key in ["accuracy", "precision", "recall", "f1", "auroc", "flagged_rate"]
    }
    tokens = RJ.token_accounting(records)
    cost = RJ.estimate_cost(tokens, backend)
    latency = RJ.latency_summary(records)
    errors = records[records["error"].notna()] if "error" in records else records.iloc[0:0]
    return {
        "model": model,
        "backend": backend,
        "per_seed": per_seed,
        "mean_across_seeds": mean_across_seeds,
        "tokens": tokens,
        "cost": cost,
        "latency": latency,
        "n_failed_calls": int(len(errors)),
        "first_errors": errors["error"].head(3).tolist() if not errors.empty else [],
        "n_unique_rows": int(ok["sample_id"].nunique()),
        "n_predictions": int(len(ok)),
    }


def agreement_within_seeds(records: pd.DataFrame, seeds: list[int]) -> dict[str, Any]:
    ok = records[records["judgment"].notna()].copy()
    ok["pred"] = (ok["judgment"] == "hallucinated").astype(int)
    pivot = ok.pivot_table(index="sample_id", columns="seed", values="pred", aggfunc="first")
    pivot = pivot.dropna(subset=[s for s in seeds if s in pivot.columns])
    out: dict[str, Any] = {"n_common_rows": int(len(pivot)), "pairwise": {}}
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            if a in pivot.columns and b in pivot.columns:
                out["pairwise"][f"{a}_vs_{b}"] = _pairwise(pivot[a], pivot[b])
    kappas = [v["cohen_kappa"] for v in out["pairwise"].values() if v["cohen_kappa"] is not None]
    out["mean_pairwise_cohen_kappa"] = float(np.mean(kappas)) if kappas else None
    if len(seeds) >= 2 and all(s in pivot.columns for s in seeds):
        matrix = pivot[seeds].to_numpy(dtype=int)
        out["three_way_exact_match"] = float(np.mean(np.all(matrix == matrix[:, [0]], axis=1)))
        out["majority_vote"] = {
            "n": int(len(pivot)),
            "flagged_rate": float(_majority_vote(pivot[seeds]).mean()),
        }
    return out


def agreement_between_models(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    return _pairwise(a, b)


# ---------------------------------------------------------------------------
# Comparisons on the length-matched rows
# ---------------------------------------------------------------------------


def load_ec_xgb_scores(matched: pd.DataFrame) -> dict[int, pd.DataFrame]:
    preds = pd.read_parquet(FIXED_HP_PREDICTIONS)
    preds = preds[(preds["dataset"] == "halueval_test") & (preds["variant"] == "m3")].copy()
    out: dict[int, pd.DataFrame] = {}
    ids = set(matched["sample_id"])
    for seed, group in preds.groupby("seed"):
        group = group[group["sample_id"].isin(ids)].sort_values("sample_id").reset_index(drop=True)
        if len(group) != len(matched):
            raise RuntimeError(f"EC-XGB seed {seed} covers {len(group)} of {len(matched)} matched rows")
        out[int(seed)] = group[["sample_id", "label", "score"]].copy()
    return out


def length_only_scores(matched: pd.DataFrame) -> pd.DataFrame:
    features = pd.read_parquet(
        FEATURES_PATH, columns=["sample_id", "label", "split", "n_chars", "n_words"]
    )
    train = features[features["split"] == "train"].copy()
    if train.empty:
        raise RuntimeError("features_full.parquet has no train rows")
    cols = ["n_chars", "n_words"]
    x_train = np.column_stack([np.log1p(train[c].to_numpy(dtype=float)) for c in cols])
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=2000).fit(scaler.transform(x_train), train["label"].to_numpy(dtype=int))
    sub = matched.drop(columns=["n_words"]).merge(
        features[["sample_id", "n_chars", "n_words"]], on="sample_id", how="left", validate="one_to_one"
    )
    if sub[cols].isna().any().any():
        raise RuntimeError("missing length features for matched rows")
    x_sub = np.column_stack([np.log1p(sub[c].to_numpy(dtype=float)) for c in cols])
    sub["score"] = model.predict_proba(scaler.transform(x_sub))[:, 1]
    return sub[["sample_id", "label", "score", "n_words"]].sort_values("sample_id").reset_index(drop=True)


def _f1_binary(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def group_bootstrap_delta(
    y: np.ndarray,
    pred_a: np.ndarray,
    score_a: np.ndarray,
    pred_b: np.ndarray,
    score_b: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired item bootstrap for F1 and AUROC deltas (A minus B)."""
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in unique}
    rng = np.random.default_rng(seed)
    f1_deltas: list[float] = []
    auc_deltas: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index[g] for g in chosen])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        pa, pb = pred_a[idx], pred_b[idx]
        sa, sb = score_a[idx], score_b[idx]
        f1_deltas.append(_f1_binary(yy, pa) - _f1_binary(yy, pb))
        auc_deltas.append(float(roc_auc_score(yy, sa) - roc_auc_score(yy, sb)))
    return {
        "f1_delta": float(np.mean(f1_deltas)) if f1_deltas else None,
        "f1_ci95": [float(np.percentile(f1_deltas, 2.5)), float(np.percentile(f1_deltas, 97.5))] if f1_deltas else None,
        "auroc_delta": float(np.mean(auc_deltas)) if auc_deltas else None,
        "auroc_ci95": [float(np.percentile(auc_deltas, 2.5)), float(np.percentile(auc_deltas, 97.5))] if auc_deltas else None,
        "n_resamples": len(f1_deltas),
    }


# ---------------------------------------------------------------------------
# Experiment 1: length-matched judge
# ---------------------------------------------------------------------------


def run_length_matched(backends: list[str], budget_usd: float, fresh: bool = False) -> dict[str, Any]:
    matched = load_length_matched()
    results: dict[str, Any] = {}
    prior_spend = existing_p0_spend()
    logger.info("Existing P0 OpenAI checkpoint spend: $%.4f", prior_spend)

    if "openai" in backends:
        model = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
        records = run_calls(
            matched,
            experiment="length_matched",
            backend="openai",
            seeds=GPT_SEEDS,
            prompt_name="current",
            concurrency=GPT_CONCURRENCY,
            model=model,
            url=RJ.QWEN_URL,
            max_completion_tokens=RJ.MAX_COMPLETION_TOKENS,
            max_predict=RJ.QWEN_MAX_PREDICT,
            fresh=fresh,
            budget_usd=budget_usd,
            spent_usd=prior_spend,
        )
        results["gpt"] = build_judge_metrics(
            records, rows=matched, seeds=GPT_SEEDS, backend="openai", model=model
        )
        results["gpt"]["agreement"] = agreement_within_seeds(records, GPT_SEEDS)
        results["gpt_records"] = records

    if "qwen" in backends:
        model = os.environ.get("QWEN_MODEL", RJ.QWEN_MODEL)
        records = run_calls(
            matched,
            experiment="length_matched",
            backend="qwen",
            seeds=QWEN_SEEDS,
            prompt_name="current",
            concurrency=QWEN_CONCURRENCY,
            model=model,
            url=RJ.QWEN_URL,
            max_completion_tokens=RJ.MAX_COMPLETION_TOKENS,
            max_predict=RJ.QWEN_MAX_PREDICT,
            fresh=fresh,
            budget_usd=None,
            spent_usd=0.0,
        )
        results["qwen"] = build_judge_metrics(
            records, rows=matched, seeds=QWEN_SEEDS, backend="qwen", model=model
        )
        results["qwen_records"] = records

    return _finalize_length_matched(matched, results, backends)


def _finalize_length_matched(matched: pd.DataFrame, results: dict[str, Any], backends: list[str]) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y = matched["label"].to_numpy(dtype=int)
    groups = matched["item_idx"].to_numpy()

    baselines: dict[str, Any] = {}
    ec = load_ec_xgb_scores(matched)
    ec_per_seed = {}
    for seed, frame in sorted(ec.items()):
        frame = frame.merge(matched[["sample_id", "item_idx"]], on="sample_id", how="left", validate="one_to_one")
        block = baseline_metric_block(
            frame["label"].to_numpy(dtype=int),
            frame["score"].to_numpy(dtype=float),
            frame["item_idx"].to_numpy(),
        )
        ec_per_seed[str(seed)] = block
    baselines["ec_xgb_m3"] = {
        "model": "EC-XGB (fixed-hp m3)",
        "per_seed": ec_per_seed,
        "mean_across_seeds": {
            key: RJ._mean_sd([ec_per_seed[str(s)].get(key) for s in GPT_SEEDS])
            for key in ["accuracy", "precision", "recall", "f1", "auroc", "flagged_rate"]
        },
    }

    length = length_only_scores(matched)
    length = length.merge(matched[["sample_id", "item_idx", "label"]], on="sample_id", how="left", validate="one_to_one", suffixes=("", "_row"))
    lb = baseline_metric_block(
        length["label_row"].to_numpy(dtype=int),
        length["score"].to_numpy(dtype=float),
        length["item_idx"].to_numpy(),
    )
    baselines["length_only"] = {
        "model": "Length-only logistic (log n_chars + log n_words)",
        "fit_rows": 14000,
        "threshold": 0.5,
        "metrics": lb,
    }

    # Per-row prediction table.
    pred_rows = matched[["sample_id", "item_idx", "label", "n_words", "bin"]].copy()
    ec_mean = np.mean(np.column_stack([ec[s]["score"].to_numpy(dtype=float) for s in GPT_SEEDS]), axis=1)
    ec_ids = ec[GPT_SEEDS[0]]["sample_id"].tolist()
    ec_mean_frame = pd.DataFrame({"sample_id": ec_ids, "ec_xgb_mean_score": ec_mean})
    pred_rows = pred_rows.merge(ec_mean_frame, on="sample_id", how="left", validate="one_to_one")
    pred_rows = pred_rows.merge(
        length[["sample_id", "score"]].rename(columns={"score": "length_only_score"}),
        on="sample_id", how="left", validate="one_to_one",
    )
    if "gpt_records" in results:
        gpt = results["gpt_records"]
        gpt = gpt[gpt["judgment"].notna()]
        for seed in GPT_SEEDS:
            sub = gpt[gpt["seed"] == seed][["sample_id", "judgment", "confidence", "score"]]
            sub = sub.rename(columns={
                "judgment": f"gpt_seed{seed}_judgment",
                "confidence": f"gpt_seed{seed}_confidence",
                "score": f"gpt_seed{seed}_score",
            })
            pred_rows = pred_rows.merge(sub, on="sample_id", how="left", validate="one_to_one")
    if "qwen_records" in results:
        qwen = results["qwen_records"]
        qwen = qwen[qwen["judgment"].notna()][["sample_id", "judgment", "confidence", "score"]]
        qwen = qwen.rename(columns={
            "judgment": "qwen_seed42_judgment",
            "confidence": "qwen_seed42_confidence",
            "score": "qwen_seed42_score",
        })
        pred_rows = pred_rows.merge(qwen, on="sample_id", how="left", validate="one_to_one")
    pred_rows = pred_rows.sort_values("sample_id").reset_index(drop=True)
    pred_rows.to_csv(OUT_DIR / "judge_length_matched_predictions.csv", index=False)

    # Comparison deltas on the same 158 rows, paired at the item level.
    comparisons: dict[str, Any] = {}
    ec_seed42 = ec[42].set_index("sample_id").loc[matched["sample_id"]]
    len_sorted = length.set_index("sample_id").loc[matched["sample_id"]]

    def _judge_vectors(records: pd.DataFrame, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
        ok = records[records["judgment"].notna()].copy()
        ok["pred"] = (ok["judgment"] == "hallucinated").astype(int)
        ok["score"] = ok["score"].astype(float)
        pivot_pred = ok.pivot_table(index="sample_id", columns="seed", values="pred", aggfunc="first")
        pivot_score = ok.pivot_table(index="sample_id", columns="seed", values="score", aggfunc="first")
        pivot_pred = pivot_pred.loc[matched["sample_id"], seeds]
        pivot_score = pivot_score.loc[matched["sample_id"], seeds]
        if len(seeds) > 1:
            return _majority_vote(pivot_pred).to_numpy(dtype=int), pivot_score.mean(axis=1).to_numpy(dtype=float)
        return pivot_pred[seeds[0]].to_numpy(dtype=int), pivot_score[seeds[0]].to_numpy(dtype=float)

    if "gpt_records" in results:
        gpt_pred, gpt_score = _judge_vectors(results["gpt_records"], GPT_SEEDS)
        comparisons["gpt_majority_vs_ec_xgb_mean"] = group_bootstrap_delta(
            y, gpt_pred, gpt_score,
            (ec_mean >= 0.5).astype(int), ec_mean, groups,
        )
        comparisons["gpt_majority_vs_length_only"] = group_bootstrap_delta(
            y, gpt_pred, gpt_score,
            (len_sorted["score"].to_numpy(dtype=float) >= 0.5).astype(int),
            len_sorted["score"].to_numpy(dtype=float), groups,
        )
        # GPT seed 42 against EC-XGB seed 42.
        gpt42 = results["gpt_records"][results["gpt_records"]["seed"] == 42].set_index("sample_id").loc[matched["sample_id"]]
        comparisons["gpt_seed42_vs_ec_xgb_seed42"] = group_bootstrap_delta(
            y, (gpt42["judgment"] == "hallucinated").astype(int).to_numpy(dtype=int), gpt42["score"].to_numpy(dtype=float),
            (ec_seed42["score"].to_numpy(dtype=float) >= 0.5).astype(int), ec_seed42["score"].to_numpy(dtype=float), groups,
        )
        # Agreement with baselines, same seed 42.
        comparisons["agreement_gpt_seed42_vs_ec_xgb_seed42"] = agreement_between_models(
            pd.Series((gpt42["judgment"] == "hallucinated").astype(int).to_numpy(), index=matched["sample_id"]),
            pd.Series((ec_seed42["score"].to_numpy(dtype=float) >= 0.5).astype(int), index=matched["sample_id"]),
        )
        comparisons["agreement_gpt_seed42_vs_length_only"] = agreement_between_models(
            pd.Series((gpt42["judgment"] == "hallucinated").astype(int).to_numpy(), index=matched["sample_id"]),
            pd.Series((len_sorted["score"].to_numpy(dtype=float) >= 0.5).astype(int), index=matched["sample_id"]),
        )
    if "qwen_records" in results:
        qwen_pred, qwen_score = _judge_vectors(results["qwen_records"], QWEN_SEEDS)
        comparisons["qwen_seed42_vs_ec_xgb_seed42"] = group_bootstrap_delta(
            y, qwen_pred, qwen_score,
            (ec_seed42["score"].to_numpy(dtype=float) >= 0.5).astype(int), ec_seed42["score"].to_numpy(dtype=float), groups,
        )
        comparisons["qwen_seed42_vs_length_only"] = group_bootstrap_delta(
            y, qwen_pred, qwen_score,
            (len_sorted["score"].to_numpy(dtype=float) >= 0.5).astype(int),
            len_sorted["score"].to_numpy(dtype=float), groups,
        )
        comparisons["agreement_qwen_seed42_vs_ec_xgb_seed42"] = agreement_between_models(
            pd.Series(qwen_pred, index=matched["sample_id"]),
            pd.Series((ec_seed42["score"].to_numpy(dtype=float) >= 0.5).astype(int), index=matched["sample_id"]),
        )
        comparisons["agreement_qwen_seed42_vs_length_only"] = agreement_between_models(
            pd.Series(qwen_pred, index=matched["sample_id"]),
            pd.Series((len_sorted["score"].to_numpy(dtype=float) >= 0.5).astype(int), index=matched["sample_id"]),
        )
    if "gpt_records" in results and "qwen_records" in results:
        gpt42_pred, _ = _judge_vectors(results["gpt_records"], [42])
        qwen_pred, _ = _judge_vectors(results["qwen_records"], [42])
        comparisons["agreement_gpt_seed42_vs_qwen_seed42"] = agreement_between_models(
            pd.Series(gpt42_pred, index=matched["sample_id"]),
            pd.Series(qwen_pred, index=matched["sample_id"]),
        )
    # EC-XGB vs length-only reference delta.
    comparisons["ec_xgb_mean_vs_length_only"] = group_bootstrap_delta(
        y, (ec_mean >= 0.5).astype(int), ec_mean,
        (len_sorted["score"].to_numpy(dtype=float) >= 0.5).astype(int),
        len_sorted["score"].to_numpy(dtype=float), groups,
    )

    # Metric rows.
    rows_out: list[dict[str, Any]] = []
    if "gpt" in results:
        gpt = results["gpt"]
        gpt_records = results["gpt_records"]
        for seed in GPT_SEEDS:
            block = gpt["per_seed"].get(str(seed))
            if not block:
                continue
            seed_tokens = RJ.token_accounting(gpt_records[gpt_records["seed"] == seed])
            seed_cost = RJ.estimate_cost(seed_tokens, "openai")
            rows_out.append(_metric_row(
                "llm_judge", gpt["model"], seed, block,
                {"tokens": seed_tokens, "cost": seed_cost},
                note="current WP6 prompt; flag = judgment == hallucinated",
            ))
    if "qwen" in results:
        qwen = results["qwen"]
        qwen_records = results["qwen_records"]
        for seed in QWEN_SEEDS:
            block = qwen["per_seed"].get(str(seed))
            if not block:
                continue
            seed_tokens = RJ.token_accounting(qwen_records[qwen_records["seed"] == seed])
            seed_cost = RJ.estimate_cost(seed_tokens, "qwen")
            rows_out.append(_metric_row(
                "llm_judge", qwen["model"], seed, block,
                {"tokens": seed_tokens, "cost": seed_cost},
                note="current WP6 prompt; flag = judgment == hallucinated",
            ))
    for seed in GPT_SEEDS:
        block = ec_per_seed[str(seed)]
        rows_out.append(_metric_row("ec_xgb_m3", "EC-XGB fixed-hp m3", seed, block, None, note="raw EC-XGB probability, threshold 0.5"))
    rows_out.append(_metric_row("length_only", "Length-only logistic", "fit_once", lb, None, note="fit on HaluEval train; threshold 0.5"))
    csv_frame = pd.DataFrame(rows_out)
    csv_frame.to_csv(OUT_DIR / "judge_length_matched.csv", index=False)

    total_tokens = {}
    total_cost = {"usd": 0.0}
    for key in ["gpt", "qwen"]:
        if key in results:
            total_tokens[key] = results[key]["tokens"]
            total_cost[key] = results[key]["cost"]
            total_cost["usd"] += float(results[key]["cost"].get("usd") or 0.0)
    total_cost["pricing"] = RJ.PRICING

    payload = {
        "meta": {
            "experiment": "p0_judge_length_matched",
            "date_utc": _now(),
            "branch": "rebuild",
            "dataset": "HaluEval test, exact-length-matched subset",
            "n_rows": int(len(matched)),
            "positives": int(y.sum()),
            "negatives": int((y == 0).sum()),
            "prevalence": float(y.mean()),
            "n_items": int(matched["item_idx"].nunique()),
            "matched_ids_file": str(MATCHED_IDS_PATH.relative_to(ROOT)),
            "matched_summary_file": "artifacts/results/rebuild/length_control_fixed_hp_summary.csv",
            "prompt_variant": "current",
            "prompt_sha256": PROMPT_VARIANTS["current"]["sha256"],
            "system_prompt_tokens_o200k": PROMPT_VARIANTS["current"]["tokens_o200k"],
            "prompt_doc": "docs/rebuild/p0/prompt_variants.md",
            "gpt_model": results.get("gpt", {}).get("model"),
            "gpt_seeds": GPT_SEEDS,
            "gpt_concurrency": GPT_CONCURRENCY,
            "qwen_model": results.get("qwen", {}).get("model"),
            "qwen_seeds": QWEN_SEEDS,
            "qwen_concurrency": QWEN_CONCURRENCY,
            "decoding": {
                "openai": "chat completions, max_completion_tokens=512 (auto-double to 2048), seed passed, response_format json_object, no temperature",
                "qwen": "Ollama /api/chat, temperature 0, num_predict=300, num_ctx=8192, think false, seed passed",
            },
            "operating_point": "EC-XGB and length-only use score >= 0.5; judges use the model's own grounded/hallucinated judgment for flag rate and F1, and the confidence-derived score for AUROC",
            "bootstrap": f"item-level bootstrap, {N_BOOTSTRAP} resamples, seed {BOOTSTRAP_SEED}",
            "note": "No human evaluation. The subset is balanced 79/79 by construction, so prevalence is 0.5 and is not the natural HaluEval prevalence.",
        },
        "models": {
            "gpt": {k: v for k, v in results.get("gpt", {}).items() if k != "gpt_records"},
            "qwen": {k: v for k, v in results.get("qwen", {}).items() if k != "qwen_records"},
            "ec_xgb_m3": baselines["ec_xgb_m3"],
            "length_only": baselines["length_only"],
        },
        "comparisons": comparisons,
        "tokens": total_tokens,
        "cost": total_cost,
    }
    _write_json(OUT_DIR / "judge_length_matched.json", payload)
    _write_length_report(payload)
    return payload


def _metric_row(
    family: str,
    model: str,
    seed: Any,
    block: dict[str, Any],
    tokens_source: dict[str, Any] | None,
    *,
    note: str,
) -> dict[str, Any]:
    tokens = (tokens_source or {}).get("tokens", {}) if tokens_source else {}
    cost = (tokens_source or {}).get("cost", {}) if tokens_source else {}
    return {
        "model_family": family,
        "model": model,
        "seed": seed,
        "n": block.get("n"),
        "positives": block.get("positives", int(round((block.get("prevalence") or 0) * (block.get("n") or 0)))),
        "prevalence": _round(block.get("prevalence")),
        "accuracy": _round(block.get("accuracy")),
        "precision": _round(block.get("precision")),
        "recall": _round(block.get("recall")),
        "f1": _round(block.get("f1")),
        "auroc": _round(block.get("auroc")),
        "flagged_rate": _round(block.get("flagged_rate")),
        "f1_ci_low": _round(_ci(block, "f1", 0)),
        "f1_ci_high": _round(_ci(block, "f1", 1)),
        "auroc_ci_low": _round(_ci(block, "auroc", 0)),
        "auroc_ci_high": _round(_ci(block, "auroc", 1)),
        "parse_failures": block.get("parse_failures"),
        "prompt_tokens": tokens.get("prompt_tokens"),
        "cached_prompt_tokens": tokens.get("cached_prompt_tokens"),
        "completion_tokens": tokens.get("completion_tokens"),
        "cost_usd": _round((cost or {}).get("usd"), 8),
        "note": note,
    }


# ---------------------------------------------------------------------------
# Experiment 2: prompt sensitivity
# ---------------------------------------------------------------------------


def run_prompt_sensitivity(backends: list[str], budget_usd: float, fresh: bool = False) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = select_prompt_rows(seed=42, n_items=50)
    rows[["sample_id", "item_idx", "label", "question", "answer"]].to_csv(
        OUT_DIR / "prompt_sensitivity_rows.csv", index=False
    )
    results: dict[str, dict[str, pd.DataFrame]] = {"gpt": {}, "qwen": {}}
    prior_spend = existing_p0_spend()
    logger.info("Existing P0 OpenAI checkpoint spend: $%.4f", prior_spend)

    if "openai" in backends:
        model = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
        for prompt_name in PROMPT_VARIANTS:
            # Recompute after each prompt so the cap covers the whole P0 run.
            prior_spend = existing_p0_spend()
            results["gpt"][prompt_name] = run_calls(
                rows,
                experiment="prompt_sensitivity",
                backend="openai",
                seeds=GPT_SEEDS,
                prompt_name=prompt_name,
                concurrency=GPT_CONCURRENCY,
                model=model,
                url=RJ.QWEN_URL,
                max_completion_tokens=RJ.MAX_COMPLETION_TOKENS,
                max_predict=RJ.QWEN_MAX_PREDICT,
                fresh=fresh,
                budget_usd=budget_usd,
                spent_usd=prior_spend,
            )

    if "qwen" in backends:
        model = os.environ.get("QWEN_MODEL", RJ.QWEN_MODEL)
        for prompt_name in PROMPT_VARIANTS:
            results["qwen"][prompt_name] = run_calls(
                rows,
                experiment="prompt_sensitivity",
                backend="qwen",
                seeds=QWEN_SEEDS,
                prompt_name=prompt_name,
                concurrency=QWEN_CONCURRENCY,
                model=model,
                url=RJ.QWEN_URL,
                max_completion_tokens=RJ.MAX_COMPLETION_TOKENS,
                max_predict=RJ.QWEN_MAX_PREDICT,
                fresh=fresh,
                budget_usd=None,
                spent_usd=0.0,
            )

    return _finalize_prompt_sensitivity(rows, results, backends)


def _prompt_metrics_for(
    records: pd.DataFrame,
    rows: pd.DataFrame,
    seeds: list[int],
    backend: str,
    model: str,
    prompt_name: str,
) -> dict[str, Any]:
    block = build_judge_metrics(records, rows=rows, seeds=seeds, backend=backend, model=model)
    block["prompt_variant"] = prompt_name
    block["prompt_sha256"] = PROMPT_VARIANTS[prompt_name]["sha256"]
    return block


def _finalize_prompt_sensitivity(
    rows: pd.DataFrame,
    results: dict[str, dict[str, pd.DataFrame]],
    backends: list[str],
) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y = rows["label"].to_numpy(dtype=int)

    metrics: dict[str, dict[str, Any]] = {"gpt": {}, "qwen": {}}
    if "gpt" in results:
        model = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
        for prompt_name, records in results["gpt"].items():
            metrics["gpt"][prompt_name] = _prompt_metrics_for(
                records, rows, GPT_SEEDS, "openai", model, prompt_name
            )
            metrics["gpt"][prompt_name]["agreement_within_seeds"] = agreement_within_seeds(records, GPT_SEEDS)
    if "qwen" in results:
        model = os.environ.get("QWEN_MODEL", RJ.QWEN_MODEL)
        for prompt_name, records in results["qwen"].items():
            metrics["qwen"][prompt_name] = _prompt_metrics_for(
                records, rows, QWEN_SEEDS, "qwen", model, prompt_name
            )

    # CSV of per-run metrics.
    csv_rows: list[dict[str, Any]] = []
    for backend_key, block_set in metrics.items():
        for prompt_name, block in block_set.items():
            records = results.get(backend_key, {}).get(prompt_name)
            if records is None or records.empty:
                continue
            for seed in (GPT_SEEDS if backend_key == "gpt" else QWEN_SEEDS):
                seed_block = block["per_seed"].get(str(seed))
                if not seed_block:
                    continue
                seed_records = records[records["seed"] == seed]
                seed_tokens = RJ.token_accounting(seed_records)
                seed_cost = RJ.estimate_cost(seed_tokens, "openai" if backend_key == "gpt" else "qwen")
                csv_rows.append(_prompt_metric_row(
                    backend_key, block, prompt_name, seed, seed_block, seed_tokens, seed_cost
                ))
    csv_frame = pd.DataFrame(csv_rows)
    csv_frame.to_csv(OUT_DIR / "prompt_sensitivity.csv", index=False)

    # Aggregates and agreement.
    summary: dict[str, Any] = {"gpt": {}, "qwen": {}}
    if metrics.get("gpt"):
        per_prompt: dict[str, Any] = {}
        all_runs: list[dict[str, Any]] = []
        for prompt_name, block in metrics["gpt"].items():
            runs = [block["per_seed"][str(s)] for s in GPT_SEEDS if str(s) in block["per_seed"]]
            all_runs.extend(runs)
            per_prompt[prompt_name] = {
                "f1": _metric_range([r.get("f1") for r in runs]),
                "auroc": _metric_range([r.get("auroc") for r in runs]),
                "flagged_rate": _metric_range([r.get("flagged_rate") for r in runs]),
                "accuracy": _metric_range([r.get("accuracy") for r in runs]),
            }
        summary["gpt"] = {
            "by_prompt": per_prompt,
            "overall_range": {
                key: _metric_range([r.get(key) for r in all_runs])
                for key in ["f1", "auroc", "flagged_rate", "accuracy"]
            },
            "range_across_prompts_by_seed": {
                str(seed): {
                    key: _metric_range([
                        metrics["gpt"][p]["per_seed"][str(seed)].get(key)
                        for p in PROMPT_VARIANTS
                        if str(seed) in metrics["gpt"][p]["per_seed"]
                    ])
                    for key in ["f1", "auroc", "flagged_rate"]
                }
                for seed in GPT_SEEDS
            },
        }
    if metrics.get("qwen"):
        per_prompt = {}
        all_runs = []
        for prompt_name, block in metrics["qwen"].items():
            runs = [block["per_seed"][str(s)] for s in QWEN_SEEDS if str(s) in block["per_seed"]]
            all_runs.extend(runs)
            per_prompt[prompt_name] = {
                "f1": _metric_range([r.get("f1") for r in runs]),
                "auroc": _metric_range([r.get("auroc") for r in runs]),
                "flagged_rate": _metric_range([r.get("flagged_rate") for r in runs]),
                "accuracy": _metric_range([r.get("accuracy") for r in runs]),
            }
        summary["qwen"] = {
            "by_prompt": per_prompt,
            "overall_range": {
                key: _metric_range([r.get(key) for r in all_runs])
                for key in ["f1", "auroc", "flagged_rate", "accuracy"]
            },
        }

    # Pairwise agreement.
    agreement: dict[str, Any] = {}
    if results.get("gpt"):
        prompt_pairs = list(PROMPT_VARIANTS.keys())
        by_seed: dict[str, Any] = {}
        for seed in GPT_SEEDS:
            series = {}
            for prompt_name, records in results["gpt"].items():
                ok = records[(records["judgment"].notna()) & (records["seed"] == seed)].copy()
                ok["pred"] = (ok["judgment"] == "hallucinated").astype(int)
                series[prompt_name] = ok.set_index("sample_id")["pred"]
            by_seed[str(seed)] = {
                f"{a}_vs_{b}": _pairwise(series[a], series[b])
                for i, a in enumerate(prompt_pairs)
                for b in prompt_pairs[i + 1:]
            }
        seed_pairs: dict[str, Any] = {}
        for prompt_name, records in results["gpt"].items():
            seed_pairs[prompt_name] = agreement_within_seeds(records, GPT_SEEDS)
        # Three-prompt majority at each seed against the label.
        majority_vs_label: dict[str, Any] = {}
        for seed in GPT_SEEDS:
            series = {}
            for prompt_name, records in results["gpt"].items():
                ok = records[(records["judgment"].notna()) & (records["seed"] == seed)].copy()
                ok["pred"] = (ok["judgment"] == "hallucinated").astype(int)
                series[prompt_name] = ok.set_index("sample_id")["pred"]
            frame = pd.DataFrame(series).dropna()
            if len(frame) == len(rows):
                majority = _majority_vote(frame)
                labels = _labels_for(majority.index, rows)
                majority_vs_label[str(seed)] = {
                    "n": int(len(majority)),
                    "accuracy": float(np.mean(majority.to_numpy(dtype=int) == labels)),
                    "cohen_kappa": RJ._kappa(majority.to_numpy(dtype=int), labels),
                    "flagged_rate": float(majority.mean()),
                }
        agreement["gpt_prompt_pairs_by_seed"] = by_seed
        agreement["gpt_seed_pairs_by_prompt"] = seed_pairs
        agreement["gpt_three_prompt_majority_vs_label"] = majority_vs_label

    if results.get("qwen"):
        series = {}
        for prompt_name, records in results["qwen"].items():
            ok = records[records["judgment"].notna()].copy()
            ok["pred"] = (ok["judgment"] == "hallucinated").astype(int)
            series[prompt_name] = ok.set_index("sample_id")["pred"]
        prompt_pairs = list(PROMPT_VARIANTS.keys())
        agreement["qwen_prompt_pairs"] = {
            f"{a}_vs_{b}": _pairwise(series[a], series[b])
            for i, a in enumerate(prompt_pairs)
            for b in prompt_pairs[i + 1:]
        }
    if results.get("gpt") and results.get("qwen"):
        cross: dict[str, Any] = {}
        for prompt_name in PROMPT_VARIANTS:
            g = results["gpt"][prompt_name]
            g = g[(g["judgment"].notna()) & (g["seed"] == 42)].copy()
            g["pred"] = (g["judgment"] == "hallucinated").astype(int)
            q = results["qwen"][prompt_name]
            q = q[q["judgment"].notna()].copy()
            q["pred"] = (q["judgment"] == "hallucinated").astype(int)
            cross[prompt_name] = _pairwise(
                g.set_index("sample_id")["pred"], q.set_index("sample_id")["pred"]
            )
        agreement["gpt_seed42_vs_qwen_seed42_by_prompt"] = cross

    tokens = {}
    cost = {"usd": 0.0}
    for key in ["gpt", "qwen"]:
        if metrics.get(key):
            calls = pd.concat(
                [records for records in results[key].values() if not records.empty],
                ignore_index=True,
            ) if results[key] else pd.DataFrame()
            tokens[key] = RJ.token_accounting(calls) if not calls.empty else {"calls": 0}
            cost[key] = RJ.estimate_cost(tokens[key], "openai" if key == "gpt" else "qwen")
            cost["usd"] += float(cost[key].get("usd") or 0.0)
    cost["pricing"] = RJ.PRICING

    payload = {
        "meta": {
            "experiment": "p0_prompt_sensitivity",
            "date_utc": _now(),
            "branch": "rebuild",
            "dataset": "HaluEval test, paired label-stratified sample (50 items, 50 grounded + 50 hallucinated)",
            "n_rows": int(len(rows)),
            "positives": int(y.sum()),
            "negatives": int((y == 0).sum()),
            "prevalence": float(y.mean()),
            "n_items": int(rows["item_idx"].nunique()),
            "sampling_seed": 42,
            "sampling": "50 item_idx values drawn with numpy default_rng(42); both label variants kept per item",
            "prompt_variants": {
                name: {
                    "title": meta["title"],
                    "sha256": meta["sha256"],
                    "tokens_o200k": meta["tokens_o200k"],
                    "note": meta["note"],
                }
                for name, meta in PROMPT_VARIANTS.items()
            },
            "prompt_doc": "docs/rebuild/p0/prompt_variants.md",
            "gpt_model": metrics.get("gpt", {}).get("current", {}).get("model"),
            "gpt_seeds": GPT_SEEDS,
            "gpt_concurrency": GPT_CONCURRENCY,
            "qwen_model": metrics.get("qwen", {}).get("current", {}).get("model"),
            "qwen_seeds": QWEN_SEEDS,
            "qwen_concurrency": QWEN_CONCURRENCY,
            "decoding": {
                "openai": "chat completions, max_completion_tokens=512 (auto-double to 2048), seed passed, response_format json_object, no temperature",
                "qwen": "Ollama /api/chat, temperature 0, num_predict=300, num_ctx=8192, think false, seed passed",
            },
            "bootstrap": f"item-level bootstrap, {N_BOOTSTRAP} resamples, seed {BOOTSTRAP_SEED}",
            "selection_rule": "Prompts were fixed before evaluation. No prompt is selected using test performance.",
        },
        "runs": metrics,
        "summary": summary,
        "agreement": agreement,
        "tokens": tokens,
        "cost": cost,
    }
    _write_json(OUT_DIR / "prompt_sensitivity.json", payload)
    _write_prompt_report(payload)
    return payload


def _prompt_metric_row(
    backend_key: str,
    block: dict[str, Any],
    prompt_name: str,
    seed: int,
    seed_block: dict[str, Any],
    seed_tokens: dict[str, Any],
    seed_cost: dict[str, Any],
) -> dict[str, Any]:
    return {
        "backend": backend_key,
        "model": block.get("model"),
        "prompt_variant": prompt_name,
        "prompt_sha256": PROMPT_VARIANTS[prompt_name]["sha256"],
        "prompt_tokens_o200k": PROMPT_VARIANTS[prompt_name]["tokens_o200k"],
        "seed": seed,
        "n": seed_block.get("n"),
        "prevalence": _round(seed_block.get("prevalence")),
        "accuracy": _round(seed_block.get("accuracy")),
        "precision": _round(seed_block.get("precision")),
        "recall": _round(seed_block.get("recall")),
        "f1": _round(seed_block.get("f1")),
        "auroc": _round(seed_block.get("auroc")),
        "flagged_rate": _round(seed_block.get("flagged_rate")),
        "f1_ci_low": _round(_ci(seed_block, "f1", 0)),
        "f1_ci_high": _round(_ci(seed_block, "f1", 1)),
        "auroc_ci_low": _round(_ci(seed_block, "auroc", 0)),
        "auroc_ci_high": _round(_ci(seed_block, "auroc", 1)),
        "parse_failures": seed_block.get("parse_failures"),
        "prompt_tokens_mean": _round(seed_tokens.get("mean_prompt_tokens"), 2),
        "completion_tokens_mean": _round(seed_tokens.get("mean_completion_tokens"), 2),
        "cost_usd_run": _round(seed_cost.get("usd"), 8),
    }


# ---------------------------------------------------------------------------
# Prompt documentation
# ---------------------------------------------------------------------------


def write_prompt_doc() -> None:
    PROMPT_DOC.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# P0 judge prompt variants")
    lines.append("")
    lines.append(f"Generated: {_now()}")
    lines.append("")
    lines.append("Branch: `rebuild`")
    lines.append("")
    lines.append(
        "This document publishes the exact system prompts used in the P0 length-matched and "
        "prompt-sensitivity judge experiments. The `current` prompt is imported verbatim from "
        "`SYSTEM_PROMPT` in `src/models/rebuild_judge.py`; the other two were fixed before any "
        "evaluation and were not selected on test performance."
    )
    lines.append("")
    lines.append("## Shared request format")
    lines.append("")
    lines.append("All variants use the same user message template, with no dynamic text in the static system prefix:")
    lines.append("")
    lines.append("```text")
    lines.append(RJ.USER_TEMPLATE)
    lines.append("```")
    lines.append("")
    lines.append("Decoding settings are identical across variants:")
    lines.append("")
    lines.append("| Backend | Model | Seeds | Output cap | Settings |")
    lines.append("|---|---|---|---:|---|")
    lines.append(
        f"| `openai` | `{os.environ.get('OPENAI_MODEL', 'gpt-5.6-luna')}` | 42, 123, 456 | "
        f"{RJ.MAX_COMPLETION_TOKENS} (auto-double to 2,048) | chat completions; `response_format={{'type': 'json_object'}}`; "
        "seed passed; no `temperature` parameter |"
    )
    lines.append(
        f"| `qwen` | `{RJ.QWEN_MODEL}` | 42 | {RJ.QWEN_MAX_PREDICT} | Ollama `/api/chat`; `temperature=0`; `num_ctx=8192`; `think=false`; seed passed |"
    )
    lines.append("")
    lines.append("| Variant | o200k tokens | SHA-256 |")
    lines.append("|---|---:|---|")
    for name, meta in PROMPT_VARIANTS.items():
        lines.append(f"| `{name}` | {meta['tokens_o200k']} | `{meta['sha256']}` |")
    lines.append("")
    for name, meta in PROMPT_VARIANTS.items():
        lines.append(f"## `{name}`: {meta['title']}")
        lines.append("")
        lines.append(meta["note"])
        lines.append("")
        lines.append("```text")
        lines.append(meta["system"])
        lines.append("```")
        lines.append("")
    lines.append("## Selection rule")
    lines.append("")
    lines.append(
        "The three prompts were written and frozen before the 100-row evaluation. The sensitivity "
        "table reports every prompt and every seed. No prompt is chosen because it scores best on "
        "the test rows."
    )
    lines.append("")
    PROMPT_DOC.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", PROMPT_DOC)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_length_report(payload: dict[str, Any]) -> None:
    meta = payload["meta"]
    lines: list[str] = []
    lines.append("# P0.3 Length-matched judge evaluation")
    lines.append("")
    lines.append(f"Date: {meta['date_utc'][:10]}")
    lines.append("")
    lines.append("Branch: `rebuild`")
    lines.append("")
    lines.append("Status: complete")
    lines.append("")
    lines.append(
        "Scope: LLM judges on the 158-row exact-length-matched HaluEval subset. No human "
        "evaluation was performed and none is claimed."
    )
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(
        f"The subset comes from `artifacts/results/rebuild/length_control_fixed_hp_matched_ids.csv`: "
        f"{meta['n_rows']} rows, {meta['positives']} hallucinated and {meta['negatives']} grounded "
        f"({meta['n_items']} unique HaluEval items, prevalence {_fmt(meta['prevalence'], 3)}). Length is matched "
        "at each exact word count, so the length-only baseline cannot separate the classes by "
        "construction."
    )
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        "GPT-5.6-Luna ran with seeds 42, 123, and 456 at concurrency 6. Qwen 3.5 4B ran with seed "
        "42 at concurrency 1 through Ollama. Both used the current WP6 prompt, unchanged. Every call "
        "was checkpointed to JSONL. EC-XGB is the fixed-hyperparameter m3 model at threshold 0.5. "
        "The length-only baseline is a logistic regression on standardized log1p(n_chars) and "
        "log1p(n_words), fit on the 14,000-row HaluEval train split. Confidence intervals are "
        f"item-level bootstraps ({N_BOOTSTRAP} resamples, seed {BOOTSTRAP_SEED})."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "F1 and flagged rate use each model's own flag decision. For the judges, a flag means the "
        "judgment is `hallucinated`. For EC-XGB and length-only, a flag means score >= 0.5. AUROC "
        "uses the continuous score. GPT rows are mean +/- sd across seeds; Qwen has one seed."
    )
    lines.append("")
    lines.append("| Model | Seeds | Accuracy | Precision | Recall | F1 | AUROC | Flagged rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    gpt = payload["models"].get("gpt", {})
    if gpt:
        mean = gpt["mean_across_seeds"]
        lines.append(
            f"| GPT-5.6-Luna | 42/123/456 | {_fmt(mean['accuracy']['mean'])} +/- {_fmt(mean['accuracy']['sd'])} | "
            f"{_fmt(mean['precision']['mean'])} +/- {_fmt(mean['precision']['sd'])} | "
            f"{_fmt(mean['recall']['mean'])} +/- {_fmt(mean['recall']['sd'])} | "
            f"{_fmt(mean['f1']['mean'])} +/- {_fmt(mean['f1']['sd'])} | "
            f"{_fmt(mean['auroc']['mean'])} +/- {_fmt(mean['auroc']['sd'])} | "
            f"{_fmt(mean['flagged_rate']['mean'])} +/- {_fmt(mean['flagged_rate']['sd'])} |"
        )
    qwen = payload["models"].get("qwen", {})
    if qwen:
        block = qwen["per_seed"].get("42", {})
        lines.append(
            f"| Qwen 3.5 4B | 42 | {_fmt(block.get('accuracy'))} | {_fmt(block.get('precision'))} | "
            f"{_fmt(block.get('recall'))} | {_fmt(block.get('f1'))} | {_fmt(block.get('auroc'))} | "
            f"{_fmt(block.get('flagged_rate'))} |"
        )
    ec = payload["models"]["ec_xgb_m3"]
    ec_mean = ec["mean_across_seeds"]
    lines.append(
        f"| EC-XGB (m3) | 42/123/456 | {_fmt(ec_mean['accuracy']['mean'])} +/- {_fmt(ec_mean['accuracy']['sd'])} | "
        f"{_fmt(ec_mean['precision']['mean'])} +/- {_fmt(ec_mean['precision']['sd'])} | "
        f"{_fmt(ec_mean['recall']['mean'])} +/- {_fmt(ec_mean['recall']['sd'])} | "
        f"{_fmt(ec_mean['f1']['mean'])} +/- {_fmt(ec_mean['f1']['sd'])} | "
        f"{_fmt(ec_mean['auroc']['mean'])} +/- {_fmt(ec_mean['auroc']['sd'])} | "
        f"{_fmt(ec_mean['flagged_rate']['mean'])} +/- {_fmt(ec_mean['flagged_rate']['sd'])} |"
    )
    lb = payload["models"]["length_only"]["metrics"]
    lines.append(
        f"| Length-only | fit once | {_fmt(lb.get('accuracy'))} | {_fmt(lb.get('precision'))} | "
        f"{_fmt(lb.get('recall'))} | {_fmt(lb.get('f1'))} | {_fmt(lb.get('auroc'))} | "
        f"{_fmt(lb.get('flagged_rate'))} |"
    )
    lines.append("")
    lines.append("## Per-seed results")
    lines.append("")
    lines.append("| Model | Seed | F1 [95% CI] | AUROC [95% CI] | Flagged rate |")
    lines.append("|---|---:|---|---|---:|")
    for key, label in [("gpt", "GPT-5.6-Luna"), ("qwen", "Qwen 3.5 4B")]:
        block_set = payload["models"].get(key, {})
        for seed, block in block_set.get("per_seed", {}).items():
            f1_ci = f"[{_fmt(_ci(block, 'f1', 0))}, {_fmt(_ci(block, 'f1', 1))}]"
            auc_ci = f"[{_fmt(_ci(block, 'auroc', 0))}, {_fmt(_ci(block, 'auroc', 1))}]"
            lines.append(
                f"| {label} | {seed} | {_fmt(block.get('f1'))} {f1_ci} | "
                f"{_fmt(block.get('auroc'))} {auc_ci} | {_fmt(block.get('flagged_rate'))} |"
            )
    for seed, block in ec["per_seed"].items():
        f1_ci = f"[{_fmt(_ci(block, 'f1', 0))}, {_fmt(_ci(block, 'f1', 1))}]"
        auc_ci = f"[{_fmt(_ci(block, 'auroc', 0))}, {_fmt(_ci(block, 'auroc', 1))}]"
        lines.append(
            f"| EC-XGB (m3) | {seed} | {_fmt(block.get('f1'))} {f1_ci} | "
            f"{_fmt(block.get('auroc'))} {auc_ci} | {_fmt(block.get('flagged_rate'))} |"
        )
    lines.append(
        f"| Length-only | fit once | {_fmt(lb.get('f1'))} "
        f"[{_fmt(_ci(lb, 'f1', 0))}, {_fmt(_ci(lb, 'f1', 1))}] | "
        f"{_fmt(lb.get('auroc'))} [{_fmt(_ci(lb, 'auroc', 0))}, {_fmt(_ci(lb, 'auroc', 1))}] | "
        f"{_fmt(lb.get('flagged_rate'))} |"
    )
    lines.append("")
    lines.append("## Comparisons on the same 158 rows")
    lines.append("")
    comp = payload["comparisons"]
    lines.append("| Comparison | F1 delta [95% CI] | AUROC delta [95% CI] |")
    lines.append("|---|---|---|")
    labels = {
        "gpt_majority_vs_ec_xgb_mean": "GPT majority vote vs EC-XGB seed mean",
        "gpt_majority_vs_length_only": "GPT majority vote vs length-only",
        "gpt_seed42_vs_ec_xgb_seed42": "GPT seed 42 vs EC-XGB seed 42",
        "qwen_seed42_vs_ec_xgb_seed42": "Qwen seed 42 vs EC-XGB seed 42",
        "qwen_seed42_vs_length_only": "Qwen seed 42 vs length-only",
        "ec_xgb_mean_vs_length_only": "EC-XGB seed mean vs length-only",
    }
    for key, label in labels.items():
        block = comp.get(key)
        if not block:
            continue
        lines.append(
            f"| {label} | {_fmt(block.get('f1_delta'))} "
            f"[{_fmt((block.get('f1_ci95') or [None, None])[0])}, {_fmt((block.get('f1_ci95') or [None, None])[1])}] | "
            f"{_fmt(block.get('auroc_delta'))} "
            f"[{_fmt((block.get('auroc_ci95') or [None, None])[0])}, {_fmt((block.get('auroc_ci95') or [None, None])[1])}] |"
        )
    lines.append("")
    lines.append("Agreement on the flag decision:")
    lines.append("")
    lines.append("| Pair | n | Exact match | Cohen's kappa |")
    lines.append("|---|---:|---:|---:|")
    if "gpt" in payload["models"]:
        for pair, block in payload["models"]["gpt"].get("agreement", {}).get("pairwise", {}).items():
            lines.append(f"| GPT {pair} (seeds) | {block['n']} | {_fmt(block['exact_match'])} | {_fmt(block['cohen_kappa'])} |")
    for key, label in [
        ("agreement_gpt_seed42_vs_ec_xgb_seed42", "GPT seed 42 vs EC-XGB seed 42"),
        ("agreement_qwen_seed42_vs_ec_xgb_seed42", "Qwen seed 42 vs EC-XGB seed 42"),
        ("agreement_gpt_seed42_vs_length_only", "GPT seed 42 vs length-only"),
        ("agreement_qwen_seed42_vs_length_only", "Qwen seed 42 vs length-only"),
        ("agreement_gpt_seed42_vs_qwen_seed42", "GPT seed 42 vs Qwen seed 42"),
    ]:
        block = comp.get(key)
        if block:
            lines.append(f"| {label} | {block['n']} | {_fmt(block['exact_match'])} | {_fmt(block['cohen_kappa'])} |")
    lines.append("")
    lines.append("## Cost and tokens")
    lines.append("")
    lines.append("| Backend | Calls | Prompt tokens | Cached prompt tokens | Completion tokens | Cost (USD) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key, label in [("gpt", "GPT-5.6-Luna"), ("qwen", "Qwen 3.5 4B")]:
        tokens = payload["tokens"].get(key, {})
        if not tokens:
            continue
        lines.append(
            f"| {label} | {tokens.get('calls', 0)} | {tokens.get('prompt_tokens', 0):,} | "
            f"{tokens.get('cached_prompt_tokens', 0):,} | {tokens.get('completion_tokens', 0):,} | "
            f"{payload['cost'][key]['usd']:.4f} |"
        )
    lines.append(f"| Total API | | | | | **{payload['cost']['usd']:.4f}** |")
    lines.append("")
    lines.append("Qwen runs locally through Ollama and has no API charge. Pricing assumptions are in the JSON.")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    if gpt:
        lines.append(
            f"GPT-5.6-Luna reaches mean F1 {_fmt(gpt['mean_across_seeds']['f1']['mean'])} and mean AUROC "
            f"{_fmt(gpt['mean_across_seeds']['auroc']['mean'])} on the length-matched subset. EC-XGB reaches "
            f"mean F1 {_fmt(ec_mean['f1']['mean'])} and mean AUROC {_fmt(ec_mean['auroc']['mean'])}. The "
            f"length-only baseline falls to AUROC {_fmt(lb.get('auroc'))}, which is near chance, so the "
            "matching removed the length shortcut."
        )
        lines.append("")
    if qwen:
        lines.append(
            f"Qwen reaches F1 {_fmt(qwen['per_seed']['42'].get('f1'))} and AUROC "
            f"{_fmt(qwen['per_seed']['42'].get('auroc'))} on the same rows."
        )
        lines.append("")
    lines.append(
        "The comparison against EC-XGB is the relevant one for the paper: it asks whether a lexical "
        "judge still competes when length cannot help. The subset is balanced by construction, so the "
        "absolute F1 values are not natural-prevalence estimates and should not be compared with the "
        "RAGTruth natural-prevalence table. No unmatched HaluEval judge run was made, so this report "
        "does not estimate a length-matching drop for the judges. It measures their level on matched rows."
    )
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```powershell")
    lines.append("& .venv\\Scripts\\python.exe src\\models\\rebuild_p0_judge.py length-matched")
    lines.append("```")
    lines.append("")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "REPORT_length_matched.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", REPORT_DIR / "REPORT_length_matched.md")


def _write_prompt_report(payload: dict[str, Any]) -> None:
    meta = payload["meta"]
    lines: list[str] = []
    lines.append("# P0.4 Judge prompt sensitivity")
    lines.append("")
    lines.append(f"Date: {meta['date_utc'][:10]}")
    lines.append("")
    lines.append("Branch: `rebuild`")
    lines.append("")
    lines.append("Status: complete")
    lines.append("")
    lines.append(
        "Scope: three fixed judge prompts on 100 HaluEval test rows. No human evaluation was "
        "performed and none is claimed."
    )
    lines.append("")
    lines.append("## Data and design")
    lines.append("")
    lines.append(
        f"The sample has {meta['n_rows']} rows: {meta['positives']} hallucinated and "
        f"{meta['negatives']} grounded, from {meta['n_items']} HaluEval test items. Each selected "
        "item contributes its correct and hallucinated answer, so the two labels share the same "
        "question and context. Selection used a paired draw with seed 42."
    )
    lines.append("")
    lines.append(
        "GPT-5.6-Luna ran 3 seeds (42, 123, 456) x 3 prompts = 900 calls at concurrency 6. "
        "Qwen 3.5 4B ran 1 seed (42) x 3 prompts = 300 calls at concurrency 1. The exact prompt "
        "text, token counts, and SHA-256 hashes are in `docs/rebuild/p0/prompt_variants.md`."
    )
    lines.append("")
    lines.append("## Results by prompt and seed")
    lines.append("")
    lines.append("| Backend | Prompt | Seed | F1 | AUROC | Flagged rate | Accuracy |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for backend_key, label in [("gpt", "GPT-5.6-Luna"), ("qwen", "Qwen 3.5 4B")]:
        block_set = payload["runs"].get(backend_key, {})
        for prompt_name, block in block_set.items():
            for seed, seed_block in block.get("per_seed", {}).items():
                lines.append(
                    f"| {label} | `{prompt_name}` | {seed} | {_fmt(seed_block.get('f1'))} | "
                    f"{_fmt(seed_block.get('auroc'))} | {_fmt(seed_block.get('flagged_rate'))} | "
                    f"{_fmt(seed_block.get('accuracy'))} |"
                )
    lines.append("")
    lines.append("## Range across prompts and seeds")
    lines.append("")
    lines.append("| Backend | Metric | Min | Max | Range | Mean | SD |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for backend_key, label in [("gpt", "GPT-5.6-Luna"), ("qwen", "Qwen 3.5 4B")]:
        summary = payload["summary"].get(backend_key, {})
        overall = summary.get("overall_range", {})
        for key in ["f1", "auroc", "flagged_rate"]:
            block = overall.get(key, {})
            lines.append(
                f"| {label} | {key} | {_fmt(block.get('min'))} | {_fmt(block.get('max'))} | "
                f"{_fmt(block.get('range'))} | {_fmt(block.get('mean'))} | {_fmt(block.get('sd'))} |"
            )
    lines.append("")
    lines.append("### GPT by prompt (mean +/- sd across 3 seeds)")
    lines.append("")
    lines.append("| Prompt | F1 | AUROC | Flagged rate |")
    lines.append("|---|---:|---:|---:|")
    for prompt_name, block in payload["summary"].get("gpt", {}).get("by_prompt", {}).items():
        lines.append(
            f"| `{prompt_name}` | {_fmt(block['f1']['mean'])} +/- {_fmt(block['f1']['sd'])} | "
            f"{_fmt(block['auroc']['mean'])} +/- {_fmt(block['auroc']['sd'])} | "
            f"{_fmt(block['flagged_rate']['mean'])} +/- {_fmt(block['flagged_rate']['sd'])} |"
        )
    lines.append("")
    lines.append("### GPT range across prompts at each seed")
    lines.append("")
    lines.append("| Seed | F1 range | AUROC range | Flagged-rate range |")
    lines.append("|---:|---:|---:|---:|")
    for seed, block in payload["summary"].get("gpt", {}).get("range_across_prompts_by_seed", {}).items():
        lines.append(
            f"| {seed} | {_fmt(block['f1']['range'])} | {_fmt(block['auroc']['range'])} | "
            f"{_fmt(block['flagged_rate']['range'])} |"
        )
    lines.append("")
    lines.append("## Pairwise agreement")
    lines.append("")
    lines.append("GPT prompt pairs at each seed (exact match / Cohen's kappa):")
    lines.append("")
    lines.append("| Seed | Pair | n | Exact match | Kappa |")
    lines.append("|---:|---|---:|---:|---:|")
    for seed, pairs in payload["agreement"].get("gpt_prompt_pairs_by_seed", {}).items():
        for pair, block in pairs.items():
            lines.append(
                f"| {seed} | {pair} | {block['n']} | {_fmt(block['exact_match'])} | {_fmt(block['cohen_kappa'])} |"
            )
    lines.append("")
    lines.append("GPT seed pairs within each prompt:")
    lines.append("")
    lines.append("| Prompt | Pair | n | Exact match | Kappa |")
    lines.append("|---|---|---:|---:|---:|")
    for prompt_name, block in payload["agreement"].get("gpt_seed_pairs_by_prompt", {}).items():
        for pair, pair_block in block.get("pairwise", {}).items():
            lines.append(
                f"| `{prompt_name}` | {pair} | {pair_block['n']} | {_fmt(pair_block['exact_match'])} | "
                f"{_fmt(pair_block['cohen_kappa'])} |"
            )
    lines.append("")
    lines.append("Qwen prompt pairs:")
    lines.append("")
    lines.append("| Pair | n | Exact match | Kappa |")
    lines.append("|---|---:|---:|---:|")
    for pair, block in payload["agreement"].get("qwen_prompt_pairs", {}).items():
        lines.append(f"| {pair} | {block['n']} | {_fmt(block['exact_match'])} | {_fmt(block['cohen_kappa'])} |")
    lines.append("")
    if payload["agreement"].get("gpt_seed42_vs_qwen_seed42_by_prompt"):
        lines.append("GPT vs Qwen at seed 42:")
        lines.append("")
        lines.append("| Prompt | n | Exact match | Kappa |")
        lines.append("|---|---:|---:|---:|")
        for prompt_name, block in payload["agreement"]["gpt_seed42_vs_qwen_seed42_by_prompt"].items():
            lines.append(f"| `{prompt_name}` | {block['n']} | {_fmt(block['exact_match'])} | {_fmt(block['cohen_kappa'])} |")
        lines.append("")
    lines.append("## Cost and tokens")
    lines.append("")
    lines.append("| Backend | Calls | Prompt tokens | Cached prompt tokens | Completion tokens | Cost (USD) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key, label in [("gpt", "GPT-5.6-Luna"), ("qwen", "Qwen 3.5 4B")]:
        tokens = payload["tokens"].get(key, {})
        if not tokens:
            continue
        lines.append(
            f"| {label} | {tokens.get('calls', 0)} | {tokens.get('prompt_tokens', 0):,} | "
            f"{tokens.get('cached_prompt_tokens', 0):,} | {tokens.get('completion_tokens', 0):,} | "
            f"{payload['cost'][key]['usd']:.4f} |"
        )
    lines.append(f"| Total API | | | | | **{payload['cost']['usd']:.4f}** |")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    gpt_overall = payload["summary"].get("gpt", {}).get("overall_range", {})
    if gpt_overall:
        lines.append(
            f"Across 9 GPT runs, F1 ranges {_fmt(gpt_overall['f1']['min'])} to {_fmt(gpt_overall['f1']['max'])} "
            f"(range {_fmt(gpt_overall['f1']['range'])}) and AUROC ranges "
            f"{_fmt(gpt_overall['auroc']['min'])} to {_fmt(gpt_overall['auroc']['max'])} "
            f"(range {_fmt(gpt_overall['auroc']['range'])}). The flagged rate ranges "
            f"{_fmt(gpt_overall['flagged_rate']['min'])} to {_fmt(gpt_overall['flagged_rate']['max'])}. "
            "Seed variation within a prompt is separated from prompt variation in the tables above."
        )
    qwen_overall = payload["summary"].get("qwen", {}).get("overall_range", {})
    if qwen_overall:
        lines.append(
            f"Qwen shows F1 range {_fmt(qwen_overall['f1']['range'])} and AUROC range "
            f"{_fmt(qwen_overall['auroc']['range'])} across the three prompts at one seed."
        )
        lines.append("")
    parse_failures = {"gpt": 0, "qwen": 0}
    for backend_key in ["gpt", "qwen"]:
        for block in payload["runs"].get(backend_key, {}).values():
            for seed_block in block.get("per_seed", {}).values():
                parse_failures[backend_key] += int(seed_block.get("parse_failures") or 0)
    lines.append(
        f"Parse failures: GPT {parse_failures['gpt']} of 900 calls, Qwen {parse_failures['qwen']} of "
        "300 calls. The parser fallback keeps every completed call in the metric; no call was dropped."
    )
    lines.append("")
    lines.append(
        "All three prompts were fixed before the run. No prompt is selected because it scores best "
        "on these test rows. The report is a sensitivity check, not a prompt search."
    )
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```powershell")
    lines.append("& .venv\\Scripts\\python.exe src\\models\\rebuild_p0_judge.py prompt-sensitivity")
    lines.append("```")
    lines.append("")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "REPORT_prompt_sensitivity.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", REPORT_DIR / "REPORT_prompt_sensitivity.md")


def _write_combined_report() -> None:
    """Write the judge `REPORT.md` without touching the other P0 workstream report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    length_md = REPORT_DIR / "REPORT_length_matched.md"
    prompt_md = REPORT_DIR / "REPORT_prompt_sensitivity.md"
    lines: list[str] = []
    lines.append("# P0 judge experiments: length match and prompt sensitivity")
    lines.append("")
    lines.append(f"Date: {_now()[:10]}")
    lines.append("")
    lines.append("Branch: `rebuild`")
    lines.append("")
    lines.append(
        "Scope: P0.3 length-matched judge and P0.4 judge prompt sensitivity. The other P0 "
        "workstream (feature ablation, SHAP stability, CPU latency) keeps its own "
        "`../REPORT.md`; this file does not modify it."
    )
    lines.append("")
    lines.append(
        "Detailed reports are `REPORT_length_matched.md` and `REPORT_prompt_sensitivity.md`; "
        "the tables below are the same verified numbers."
    )
    lines.append("")
    spend = 0.0
    gpt_calls = 0
    qwen_calls = 0
    for name in ["judge_length_matched.json", "prompt_sensitivity.json"]:
        path = OUT_DIR / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        spend += float((data.get("cost") or {}).get("usd") or 0.0)
        tokens = data.get("tokens") or {}
        gpt_calls += int(((tokens.get("gpt") or {}).get("calls")) or 0)
        qwen_calls += int(((tokens.get("qwen") or {}).get("calls")) or 0)
    lines.append(
        f"Combined OpenAI API spend for these two experiments: ${spend:.4f} across {gpt_calls:,} GPT "
        f"calls. Qwen ran locally ({qwen_calls:,} calls, $0). The P0 cap is $1.00."
    )
    lines.append("")
    lines.append("## P0.3 Length-matched judge")
    lines.append("")
    if length_md.exists():
        body = length_md.read_text(encoding="utf-8").splitlines()
        start = next((i for i, line in enumerate(body) if line.startswith("## Data")), 0)
        lines.extend(("#" + line if line.startswith("#") else line) for line in body[start:])
    else:
        lines.append("Not run yet.")
    lines.append("")
    lines.append("## P0.4 Judge prompt sensitivity")
    lines.append("")
    if prompt_md.exists():
        body = prompt_md.read_text(encoding="utf-8").splitlines()
        start = next((i for i, line in enumerate(body) if line.startswith("## Data and design")), 0)
        lines.extend(("#" + line if line.startswith("#") else line) for line in body[start:])
    else:
        lines.append("Not run yet.")
    lines.append("")
    (REPORT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", REPORT_DIR / "REPORT.md")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0 judge experiments on HaluEval")
    parser.add_argument(
        "command",
        choices=["length-matched", "prompt-sensitivity", "all", "prompt-doc", "report"],
    )
    parser.add_argument("--backends", default="openai,qwen", help="comma-separated backends (default both)")
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD, help="P0 OpenAI spend cap in USD")
    parser.add_argument("--fresh", action="store_true", help="ignore existing JSONL checkpoints")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    backends = [part.strip() for part in args.backends.split(",") if part.strip()]
    if not backends or any(b not in {"openai", "qwen"} for b in backends):
        raise SystemExit("--backends must be a comma-separated subset of openai,qwen")

    if args.command == "prompt-doc":
        write_prompt_doc()
        return
    if args.command == "report":
        length_json = OUT_DIR / "judge_length_matched.json"
        prompt_json = OUT_DIR / "prompt_sensitivity.json"
        if length_json.exists():
            _write_length_report(json.loads(length_json.read_text(encoding="utf-8")))
        if prompt_json.exists():
            _write_prompt_report(json.loads(prompt_json.read_text(encoding="utf-8")))
        _write_combined_report()
        return

    write_prompt_doc()
    if args.command in {"length-matched", "all"}:
        run_length_matched(backends, budget_usd=args.budget_usd, fresh=args.fresh)
    if args.command in {"prompt-sensitivity", "all"}:
        run_prompt_sensitivity(backends, budget_usd=args.budget_usd, fresh=args.fresh)
    _write_combined_report()

    # Final spend across all P0 OpenAI checkpoints, for the run log.
    total = existing_p0_spend()
    logger.info("P0 OpenAI checkpoint spend (retry-inclusive upper bound): $%.4f", total)
    if total > args.budget_usd:
        logger.error("P0 spend $%.4f exceeds the $%.2f cap", total, args.budget_usd)


if __name__ == "__main__":
    main()
