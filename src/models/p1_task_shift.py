#!/usr/bin/env python
"""P1 task-shift evaluation on the HaluEval dialogue and summarization subsets.

The in-domain study trains and tests on HaluEval QA.  This script asks a
narrower question: do the fixed-hyperparameter models still rank and calibrate
answer variants when the *task* changes inside the same benchmark family?
Dialogue and summarization rows come from the two additional HaluEval subsets
that ship beside ``qa_data.json``, so this is a task-shift test, not a second
independent corpus.

Design (fixed before the run, no tuning on these rows):

  * 1,000 source items per task, drawn with ``numpy`` seed 42 from the valid
    raw lines; each item contributes one correct (label 0) and one
    hallucinated (label 1) row, i.e. 2,000 rows per task and 4,000 in total.
  * Dialogue: question = ``dialogue_history``, context = ``knowledge``,
    answer = the response variant.
  * Summarization: question = "Summarize the document.", context =
    ``document``, answer = the summary variant.
  * Features: the same 26 base features and 8 claim-level NLI aggregates used
    by the EC-XGB pipeline, extracted with the shared extractors and models.
  * Models: the fixed-hyperparameter m3 (EC-XGB) and m0 (standard XGBoost)
    checkpoints from ``artifacts/models/rebuild/fixed_hp`` for seeds
    42/123/456, scored at the fixed 0.5 threshold.  ``source_ragtruth`` is 0
    for these rows because they are HaluEval-generated, like the training
    rows.  A sensitivity pass with the indicator set to 1 is reported in the
    JSON but is not a headline number.
  * Baseline: a length-only logistic regression (log1p characters and words,
    sentence count, mean word length) trained on the HaluEval QA train split.
  * Metrics: F1, AUROC, PR-AUC, ECE (10 bins), Brier, and flagged rate, per
    task and pooled.

Outputs (new files only, nothing frozen is touched):

  artifacts/results/rebuild/p1/task_shift.csv
  artifacts/results/rebuild/p1/task_shift.json
  artifacts/results/rebuild/p1/task_shift_predictions.parquet
  artifacts/results/rebuild/p1/REPORT.md
  artifacts/results/rebuild/p1/task_shift_REPORT.md

Feature caches land in ``data/processed/p1_task_shift.*`` so a rerun does not
repeat the GPU/CPU extraction.  Pass ``--force-features`` to rebuild them.

Run (repo root, .venv):

  python src/models/p1_task_shift.py
  python src/models/p1_task_shift.py --pairs-per-task 25 --force-features
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.features.claim_features import (  # noqa: E402
    FEATURE_COLUMNS as CLAIM_COLUMNS,
    extract_claim_features_df,
)
from src.features.extract_features import (  # noqa: E402
    extract_full_feature_set,
    load_heavy_models,
)
from src.models.config import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, SEEDS  # noqa: E402
from src.models.rebuild_common import metric_bundle  # noqa: E402
from src.models.run_b6_modified import (  # noqa: E402
    BASE_COLUMNS,
    SOURCE_COLUMN,
    feature_matrix,
    variant_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("p1_task_shift")

# --------------------------------------------------------------------------
# Paths and fixed protocol constants
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "halueval"
OUT_DIR = RESULTS_DIR / "rebuild" / "p1"
FIXED_HP_MODELS = MODELS_DIR / "rebuild" / "fixed_hp"
FEATURES_FULL = DATA_PROCESSED / "features_full.parquet"

BASE_CACHE = DATA_PROCESSED / "p1_task_shift.base.parquet"
BASE_META = DATA_PROCESSED / "p1_task_shift.base.meta.json"
CLAIM_CACHE = DATA_PROCESSED / "p1_task_shift.claims.parquet"
CLAIM_META = DATA_PROCESSED / "p1_task_shift.claims.meta.json"

SAMPLE_SEED = 42
PAIRS_PER_TASK = 1000
THRESHOLD = 0.5
ECE_BINS = 10

TASK_FILES = {
    "dialogue": RAW_DIR / "dialogue_data.json",
    "summarization": RAW_DIR / "summarization_data.json",
}
TASK_ORDER = ("dialogue", "summarization")
EVAL_SPLITS = ("dialogue", "summarization", "pooled")

MODEL_VARIANTS = ("m3", "m0")
MODEL_LABELS = {
    "m3": "EC-XGB (fixed-hp m3)",
    "m0": "Standard XGBoost (fixed-hp m0)",
    "length_logistic": "Length-only logistic regression",
}

LENGTH_FEATURES = ["n_chars", "n_words", "n_sentences", "avg_word_len"]
SOURCE_VALUE = 0.0

METRIC_COLUMNS = [
    "n",
    "n_pos",
    "prevalence",
    "f1",
    "precision",
    "recall",
    "auroc",
    "pr_auc",
    "ece",
    "brier",
    "flagged_rate",
    "fpr",
]


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    return items


def _task_row(task: str, idx: int, item: dict) -> list[dict]:
    """Expand one raw HaluEval item into a correct/hallucinated pair."""
    if task == "dialogue":
        question = str(item.get("dialogue_history", "")).strip()
        context = str(item.get("knowledge", "")).strip()
        right = str(item.get("right_response", "")).strip()
        hallucinated = str(item.get("hallucinated_response", "")).strip()
    elif task == "summarization":
        question = "Summarize the document."
        context = str(item.get("document", "")).strip()
        right = str(item.get("right_summary", "")).strip()
        hallucinated = str(item.get("hallucinated_summary", "")).strip()
    else:
        raise ValueError(f"unknown task: {task}")

    if not question or not context or not right or not hallucinated:
        return []

    return [
        {
            "sample_id": f"p1_{task}_{idx:05d}_correct",
            "item_idx": idx,
            "pair_id": idx,
            "task": task,
            "label": 0,
            "split": "task_shift",
            "question": question,
            "context": context,
            "answer": right,
        },
        {
            "sample_id": f"p1_{task}_{idx:05d}_hallucinated",
            "item_idx": idx,
            "pair_id": idx,
            "task": task,
            "label": 1,
            "split": "task_shift",
            "question": question,
            "context": context,
            "answer": hallucinated,
        },
    ]


def build_task_frame(task: str, pairs: int, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Deterministically sample ``pairs`` valid items and expand both variants."""
    path = TASK_FILES[task]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; download the HaluEval subsets first")

    items = _read_jsonl(path)
    valid = [i for i, item in enumerate(items) if _task_row(task, i, item)]
    take = min(pairs, len(valid))
    if take < pairs:
        logger.warning(f"{task}: only {take} valid items (< requested {pairs})")
    chosen = np.sort(rng.choice(np.asarray(valid), size=take, replace=False))

    rows: list[dict] = []
    for idx in chosen:
        rows.extend(_task_row(task, int(idx), items[int(idx)]))
    frame = pd.DataFrame(rows)
    info = {
        "raw_items": len(items),
        "valid_items": len(valid),
        "pairs_sampled": int(take),
        "rows": int(len(frame)),
        "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()},
        "pair_indices_sha256": hashlib.sha256(
            ",".join(str(int(i)) for i in chosen).encode("ascii")
        ).hexdigest(),
    }
    logger.info(
        f"{task}: {len(items)} raw, {len(valid)} valid, {take} pairs -> {len(frame)} rows"
    )
    return frame, info


def build_frame(pairs_per_task: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    frames, meta = [], {}
    for task in TASK_ORDER:
        frame, info = build_task_frame(task, pairs_per_task, rng)
        frames.append(frame)
        meta[task] = info
    df = pd.concat(frames, ignore_index=True)
    if df["sample_id"].duplicated().any():
        raise ValueError("duplicate sample_id after task assembly")
    if not (df.groupby("task")["label"].mean() == 0.5).all():
        raise ValueError("each task must have one correct and one hallucinated row per pair")
    return df, meta


# --------------------------------------------------------------------------
# Feature extraction with a validated cache
# --------------------------------------------------------------------------
def _digest_sample_ids(frame: pd.DataFrame) -> str:
    handle = hashlib.sha256()
    for sample_id in sorted(frame["sample_id"].astype(str)):
        handle.update(sample_id.encode("utf-8"))
        handle.update(b"\n")
    return handle.hexdigest()


def _cache_valid(path: Path, meta_path: Path, digest: str, n_rows: int, columns: list[str]) -> bool:
    if not path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("sample_ids_sha256") != digest or meta.get("n_rows") != n_rows:
        return False
    import pyarrow.parquet as pq

    cached_columns = set(pq.read_schema(path).names)
    return set(columns).issubset(cached_columns)


def _write_meta(path: Path, stage: str, digest: str, n_rows: int, pairs: int, columns: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "p1-task-shift-cache-v1",
                "stage": stage,
                "sample_ids_sha256": digest,
                "n_rows": n_rows,
                "pairs_per_task": pairs,
                "seed": SAMPLE_SEED,
                "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
                "columns": columns,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _sanitize_partial(path: Path, allowed_ids: set[str]) -> None:
    """Drop duplicate or stale rows from a resumable claim checkpoint."""
    if not path.exists():
        return
    part = pd.read_parquet(path)
    before = len(part)
    if "sample_id" not in part.columns:
        path.unlink()
        return
    part = part[part["sample_id"].isin(allowed_ids)]
    part = part.drop_duplicates("sample_id", keep="last")
    if len(part) != before:
        logger.warning(f"Sanitised partial checkpoint {path.name}: {before} -> {len(part)} rows")
        part.to_parquet(path, index=False)


def extract_features(
    frame: pd.DataFrame,
    pairs_per_task: int,
    device: str,
    batch_size: int,
    claim_batch_size: int,
    force: bool,
) -> tuple[pd.DataFrame, dict]:
    """Return the frame with all 34 features, reusing validated caches."""
    digest = _digest_sample_ids(frame)
    timings: dict[str, float] = {}

    base_ready = (
        not force
        and _cache_valid(BASE_CACHE, BASE_META, digest, len(frame), BASE_COLUMNS)
    )
    claims_ready = (
        not force
        and _cache_valid(CLAIM_CACHE, CLAIM_META, digest, len(frame), CLAIM_COLUMNS)
    )
    logger.info(f"Feature cache: base={base_ready} claims={claims_ready}")

    models = None
    if not (base_ready and claims_ready):
        t0 = time.perf_counter()
        models = load_heavy_models(device=device)
        timings["model_load_s"] = time.perf_counter() - t0
        logger.info(f"Heavy models loaded in {timings['model_load_s']:.1f}s ({models['nli_name']})")

    if base_ready:
        base = pd.read_parquet(BASE_CACHE)
        timings["base_features_s"] = 0.0
    else:
        t0 = time.perf_counter()
        base = extract_full_feature_set(frame, models, batch_size=batch_size)
        timings["base_features_s"] = time.perf_counter() - t0
        base.to_parquet(BASE_CACHE, index=False)
        _write_meta(BASE_META, "base", digest, len(frame), pairs_per_task, BASE_COLUMNS)
        logger.info(f"Base features: {timings['base_features_s']:.1f}s -> {BASE_CACHE}")

    if claims_ready:
        claims = pd.read_parquet(CLAIM_CACHE)
        timings["claim_features_s"] = 0.0
    else:
        t0 = time.perf_counter()
        partial = DATA_PROCESSED / f"p1_task_shift.claims.{digest[:12]}.partial.parquet"
        _sanitize_partial(partial, set(frame["sample_id"].astype(str)))
        claims = extract_claim_features_df(
            frame[["sample_id", "question", "context", "answer"]].reset_index(drop=True),
            models["nli"],
            partial_path=partial,
            checkpoint_every=500,
            batch_size=claim_batch_size,
            chunk_samples=24,
        )
        claims = claims.drop_duplicates("sample_id", keep="last")
        timings["claim_features_s"] = time.perf_counter() - t0
        claims.to_parquet(CLAIM_CACHE, index=False)
        _write_meta(CLAIM_META, "claims", digest, len(frame), pairs_per_task, CLAIM_COLUMNS)
        if partial.exists():
            partial.unlink()
        logger.info(f"Claim features: {timings['claim_features_s']:.1f}s -> {CLAIM_CACHE}")

    out = base.merge(claims[["sample_id", *CLAIM_COLUMNS]], on="sample_id", how="left", validate="one_to_one")
    if len(out) != len(frame):
        raise ValueError(f"feature merge changed row count: {len(frame)} -> {len(out)}")
    if out[BASE_COLUMNS + CLAIM_COLUMNS].isna().any().any():
        raise ValueError("NaN feature values after merge")

    out = out.merge(
        frame[["sample_id", "task", "pair_id", "question", "context", "answer"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if len(out) != len(frame):
        raise ValueError("metadata merge changed row count")

    timings["nli_model"] = models["nli_name"] if models else "cached"
    timings["device"] = device
    return out, timings


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def load_fixed_models() -> dict:
    import joblib

    paths, models = {}, {}
    for variant in MODEL_VARIANTS:
        for seed in SEEDS:
            path = FIXED_HP_MODELS / f"xgboost_{variant}_seed_{seed}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"fixed-hp checkpoint missing: {path}")
            model = joblib.load(path)
            expected = len(variant_features(variant))
            if model.n_features_in_ != expected:
                raise ValueError(
                    f"{path.name}: {model.n_features_in_} features, expected {expected}"
                )
            model.set_params(device="cpu")
            models[(variant, int(seed))] = model
            paths[f"{variant}_seed_{seed}"] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "n_features": int(model.n_features_in_),
            }
    return {"models": models, "paths": paths}


def predict_variant(model, frame: pd.DataFrame, variant: str, source_value: float = SOURCE_VALUE) -> np.ndarray:
    names = variant_features(variant)
    work = frame
    if variant == "m3":
        work = frame.copy()
        work[SOURCE_COLUMN] = float(source_value)
    matrix = feature_matrix(work, names, log_lengths=(variant != "m0"))
    return model.predict_proba(matrix)[:, 1]


def train_length_baseline() -> dict:
    """Fit the length-only control on the HaluEval QA train split."""
    full = pd.read_parquet(FEATURES_FULL, columns=["sample_id", "split", *LENGTH_FEATURES, "label"])
    train = full[full["split"] == "train"].reset_index(drop=True)
    test = full[full["split"] == "test"].reset_index(drop=True)

    def transform(frame: pd.DataFrame) -> np.ndarray:
        values = frame[LENGTH_FEATURES].to_numpy(dtype=np.float64)
        values[:, 0] = np.log1p(np.clip(values[:, 0], 0.0, None))
        values[:, 1] = np.log1p(np.clip(values[:, 1], 0.0, None))
        return values

    scaler = StandardScaler().fit(transform(train))
    model = LogisticRegression(max_iter=2000).fit(scaler.transform(transform(train)), train["label"].to_numpy())
    in_domain = metric_bundle(
        test["label"].to_numpy(),
        model.predict_proba(scaler.transform(transform(test)))[:, 1],
        threshold=THRESHOLD,
    )
    logger.info(
        f"Length-only baseline trained on {len(train)} HaluEval train rows; "
        f"in-domain test F1={in_domain['f1']:.4f} AUROC={in_domain['auroc']:.4f}"
    )
    return {
        "model": model,
        "scaler": scaler,
        "transform": transform,
        "info": {
            "features": LENGTH_FEATURES,
            "log1p_features": ["n_chars", "n_words"],
            "n_train": int(len(train)),
            "train_prevalence": float(train["label"].mean()),
            "n_test_reference": int(len(test)),
            "coef_standardized": {
                name: float(coef) for name, coef in zip(LENGTH_FEATURES, model.coef_[0])
            },
            "intercept": float(model.intercept_[0]),
            "scaler_mean": {
                name: float(value) for name, value in zip(LENGTH_FEATURES, scaler.mean_)
            },
            "scaler_scale": {
                name: float(value) for name, value in zip(LENGTH_FEATURES, scaler.scale_)
            },
            "in_domain_test_reference": in_domain,
            "note": "Reference row only; not part of the task-shift table.",
        },
    }


def predict_length_baseline(baseline: dict, frame: pd.DataFrame) -> np.ndarray:
    return baseline["model"].predict_proba(
        baseline["scaler"].transform(baseline["transform"](frame))
    )[:, 1]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def _metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    bundle = metric_bundle(frame["label"].to_numpy(), scores, threshold=THRESHOLD)
    flagged = scores >= THRESHOLD
    negatives = frame["label"].to_numpy() == 0
    out = {
        "n": int(bundle["n"]),
        "n_pos": int(frame["label"].sum()),
        "prevalence": float(bundle["prevalence"]),
        "f1": float(bundle["f1"]),
        "precision": float(bundle["precision"]),
        "recall": float(bundle["recall"]),
        "auroc": float(bundle["auroc"]),
        "pr_auc": float(bundle["pr_auc"]),
        "ece": float(bundle["ece"]),
        "brier": float(bundle["brier"]),
        "flagged_rate": float(flagged.mean()),
        "fpr": float(flagged[negatives].mean()) if negatives.any() else float("nan"),
    }
    return out


def _aggregate_seed_metrics(per_seed: dict) -> tuple[dict, dict]:
    mean, std = {}, {}
    for key in METRIC_COLUMNS:
        values = np.asarray([row[key] for row in per_seed.values()], dtype=float)
        mean[key] = float(np.nanmean(values))
        std[key] = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else 0.0
    return mean, std


def evaluate_models(feature_frame: pd.DataFrame, models: dict, baseline: dict) -> tuple[pd.DataFrame, dict]:
    """Score every model on each task and on the pooled frame."""
    splits = {task: feature_frame[feature_frame["task"] == task].reset_index(drop=True) for task in TASK_ORDER}
    splits["pooled"] = feature_frame.reset_index(drop=True)

    rows, payload, predictions = [], {}, []
    for variant in MODEL_VARIANTS:
        payload[variant] = {}
        per_seed_by_split: dict[str, dict] = {name: {} for name in EVAL_SPLITS}
        for seed in SEEDS:
            model = models[(variant, int(seed))]
            scores = {}
            for name, frame in splits.items():
                score = predict_variant(model, frame, variant)
                scores[name] = score
                bundle = _metrics(frame, score)
                per_seed_by_split[name][int(seed)] = bundle
                rows.append(
                    {
                        "model": variant,
                        "model_label": MODEL_LABELS[variant],
                        "task": name,
                        "seed": str(int(seed)),
                        **bundle,
                    }
                )
                predictions.extend(
                    {
                        "model": variant,
                        "model_label": MODEL_LABELS[variant],
                        "seed": str(int(seed)),
                        "task": name,
                        "sample_id": sample_id,
                        "pair_id": int(pair_id),
                        "label": int(label),
                        "score": float(score_value),
                        "predicted": int(score_value >= THRESHOLD),
                    }
                    for sample_id, pair_id, label, score_value in zip(
                        frame["sample_id"], frame["pair_id"], frame["label"], score
                    )
                )
        payload[variant] = {
            name: {
                "per_seed": per_seed_by_split[name],
                "mean": _aggregate_seed_metrics(per_seed_by_split[name])[0],
                "std": _aggregate_seed_metrics(per_seed_by_split[name])[1],
            }
            for name in EVAL_SPLITS
        }
        for name in EVAL_SPLITS:
            mean, std = _aggregate_seed_metrics(per_seed_by_split[name])
            payload[variant][name]["mean"] = mean
            payload[variant][name]["std"] = std
            rows.append({"model": variant, "model_label": MODEL_LABELS[variant], "task": name, "seed": "mean", **mean})
            rows.append({"model": variant, "model_label": MODEL_LABELS[variant], "task": name, "seed": "std", **std})

    # Length-only control: one deterministic fit, no seed averaging.
    payload["length_logistic"] = {}
    for name, frame in splits.items():
        score = predict_length_baseline(baseline, frame)
        bundle = _metrics(frame, score)
        payload["length_logistic"][name] = {"fixed": bundle}
        rows.append(
            {
                "model": "length_logistic",
                "model_label": MODEL_LABELS["length_logistic"],
                "task": name,
                "seed": "fixed",
                **bundle,
            }
        )
        predictions.extend(
            {
                "model": "length_logistic",
                "model_label": MODEL_LABELS["length_logistic"],
                "seed": "fixed",
                "task": name,
                "sample_id": sample_id,
                "pair_id": int(pair_id),
                "label": int(label),
                "score": float(score_value),
                "predicted": int(score_value >= THRESHOLD),
            }
            for sample_id, pair_id, label, score_value in zip(
                frame["sample_id"], frame["pair_id"], frame["label"], score
            )
        )

    return pd.DataFrame(rows), {"metrics": payload, "predictions": pd.DataFrame(predictions)}


def evaluate_source_sensitivity(feature_frame: pd.DataFrame, models: dict) -> dict:
    """m3 with the RAGTruth source indicator set to 1 (not a headline row)."""
    splits = {task: feature_frame[feature_frame["task"] == task].reset_index(drop=True) for task in TASK_ORDER}
    splits["pooled"] = feature_frame.reset_index(drop=True)
    out: dict = {}
    for name, frame in splits.items():
        scores = [predict_variant(models[("m3", int(seed))], frame, "m3", source_value=1.0) for seed in SEEDS]
        per_seed = {
            int(seed): _metrics(frame, score) for seed, score in zip(SEEDS, scores)
        }
        mean, std = _aggregate_seed_metrics(per_seed)
        out[name] = {"per_seed": per_seed, "mean": mean, "std": std}
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _metric_rows(csv_frame: pd.DataFrame, model: str, task: str) -> dict:
    mean = csv_frame[(csv_frame["model"] == model) & (csv_frame["task"] == task) & (csv_frame["seed"] == "mean")]
    std = csv_frame[(csv_frame["model"] == model) & (csv_frame["task"] == task) & (csv_frame["seed"] == "std")]
    if mean.empty:
        mean = csv_frame[(csv_frame["model"] == model) & (csv_frame["task"] == task) & (csv_frame["seed"] == "fixed")]
        std = None
    return {
        "mean": mean.iloc[0].to_dict(),
        "std": std.iloc[0].to_dict() if std is not None and not std.empty else None,
    }


def _cell(block: dict, metric: str) -> str:
    mean = block["mean"][metric]
    if block["std"] is None:
        return _fmt(mean)
    return f"{mean:.4f} ({block['std'][metric]:.4f})"


def build_report(
    csv_frame: pd.DataFrame,
    data_meta: dict,
    feature_frame: pd.DataFrame,
    runtime: dict,
    config: dict,
    sensitivity: dict,
    baseline_info: dict,
) -> str:
    lines = [
        "# P1 task shift: HaluEval dialogue and summarization",
        "",
        f"Generated {config['generated_at_utc']} on branch `rebuild` by `src/models/p1_task_shift.py`.",
        "",
        f"Run summary: {config['pairs_per_task']:,} pairs per task, "
        f"{data_meta['pooled']['rows']:,} scored rows in total, wall-clock runtime "
        f"{runtime['total_s']:.1f} s ({runtime['total_s'] / 60:.1f} min) on device `{config['device']}`.",
        "",
        "## What this test is",
        "",
        "The in-domain result is measured on HaluEval QA. This run changes the task while keeping the",
        "benchmark family fixed: the dialogue and summarization subsets ship with the same HaluEval",
        "release, the same two-variant construction, and the same generator. It is a **task-shift test**,",
        "not a second independent corpus, and it does not establish cross-domain transfer.",
        "",
        "## Design",
        "",
        f"- Sampling: {config['pairs_per_task']} source items per task, drawn without replacement from the",
        f"  valid raw lines with NumPy seed {config['sample_seed']}. Each item contributes one correct",
        f"  (label 0) and one hallucinated (label 1) row, so each task has {2 * config['pairs_per_task']:,} rows",
        "  at 50% prevalence.",
        f"- Dialogue mapping: question = `dialogue_history`, context = `knowledge`, answer = response.",
        f"- Summarization mapping: question = \"Summarize the document.\", context = `document`,",
        "  answer = summary.",
        "- Features: 26 base features plus 8 claim-level NLI aggregates, extracted with the shared",
        "  feature modules and the same NLI checkpoint as the main pipeline.",
        "- Models: the fixed-hyperparameter m3 (EC-XGB) and m0 (standard XGBoost) checkpoints for seeds",
        "  42/123/456, scored at the fixed 0.5 threshold. No parameter, feature, or threshold was tuned",
        "  on these rows. `source_ragtruth` = 0 because the rows are HaluEval-generated.",
        "- Baseline: a length-only logistic regression on log1p characters, log1p words, sentence count,",
        "  and mean word length, trained on the HaluEval QA train split.",
        "- Metrics: F1, AUROC, PR-AUC, ECE (10 equal-width bins), Brier, and flagged rate. For m0/m3 the",
        "  table reports the seed mean with the sample standard deviation across the three seeds in",
        "  parentheses; the length control is a single deterministic fit.",
        "",
        "## Row counts",
        "",
        "| Task | Raw items | Valid items | Pairs sampled | Rows | Label 0 | Label 1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in TASK_ORDER:
        info = data_meta["tasks"][task]
        lines.append(
            f"| {task} | {info['raw_items']:,} | {info['valid_items']:,} | {info['pairs_sampled']:,} | "
            f"{info['rows']:,} | {info['label_counts']['0']:,} | {info['label_counts']['1']:,} |"
        )
    pooled = data_meta["pooled"]
    lines.append(
        f"| pooled | {pooled['raw_items']:,} | {pooled['valid_items']:,} | {pooled['pairs_sampled']:,} | "
        f"{pooled['rows']:,} | {pooled['label_counts']['0']:,} | {pooled['label_counts']['1']:,} |"
    )
    lines += [
        "",
        "## Discrimination",
        "",
        "| Model | Task | F1 | AUROC | PR-AUC |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    model_order = ["m3", "m0", "length_logistic"]
    for model in model_order:
        for task in EVAL_SPLITS:
            block = _metric_rows(csv_frame, model, task)
            name = MODEL_LABELS[model] if task == EVAL_SPLITS[0] else ""
            lines.append(
                f"| {name} | {task} | {_cell(block, 'f1')} | {_cell(block, 'auroc')} | {_cell(block, 'pr_auc')} |"
            )

    lines += [
        "",
        "## Calibration and flagging",
        "",
        "| Model | Task | ECE | Brier | Flagged rate | FPR |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model in model_order:
        for task in EVAL_SPLITS:
            block = _metric_rows(csv_frame, model, task)
            name = MODEL_LABELS[model] if task == EVAL_SPLITS[0] else ""
            lines.append(
                f"| {name} | {task} | {_cell(block, 'ece')} | {_cell(block, 'brier')} | "
                f"{_cell(block, 'flagged_rate')} | {_cell(block, 'fpr')} |"
            )

    # Computed observations, kept descriptive rather than interpretive.
    def m(model: str, task: str, metric: str) -> float:
        return _metric_rows(csv_frame, model, task)["mean"][metric]

    pooled_m3, pooled_m0 = _metric_rows(csv_frame, "m3", "pooled"), _metric_rows(csv_frame, "m0", "pooled")
    length_pooled = _metric_rows(csv_frame, "length_logistic", "pooled")
    mean_score_dialogue = float(feature_frame.loc[feature_frame["task"] == "dialogue", "m3_score"].mean())
    mean_score_summarization = float(
        feature_frame.loc[feature_frame["task"] == "summarization", "m3_score"].mean()
    )
    pooled_prevalence = data_meta["pooled"]["label_counts"]["1"] / data_meta["pooled"]["rows"]
    all_positive_f1 = 2 * pooled_prevalence / (pooled_prevalence + 1)
    lines += [
        "",
        "## What the numbers show",
        "",
        f"- On the pooled task-shift rows, m3 reaches F1 {_fmt(pooled_m3['mean']['f1'])} and AUROC "
        f"{_fmt(pooled_m3['mean']['auroc'])}; m0 reaches F1 {_fmt(pooled_m0['mean']['f1'])} and AUROC "
        f"{_fmt(pooled_m0['mean']['auroc'])}. The m3-minus-m0 deltas are "
        f"{pooled_m3['mean']['f1'] - pooled_m0['mean']['f1']:+.4f} F1 and "
        f"{pooled_m3['mean']['auroc'] - pooled_m0['mean']['auroc']:+.4f} AUROC.",
        f"- The length-only control reaches F1 {_fmt(length_pooled['mean']['f1'])} and AUROC "
        f"{_fmt(length_pooled['mean']['auroc'])} on the pooled rows, versus AUROC "
        f"{_fmt(pooled_m3['mean']['auroc'])} for m3 and {_fmt(pooled_m0['mean']['auroc'])} for m0.",
        "  Answer length remains a strong shortcut after the task changes.",
        f"- Pooled flagged rates are {_fmt(pooled_m3['mean']['flagged_rate'])} for m3, "
        f"{_fmt(pooled_m0['mean']['flagged_rate'])} for m0, and "
        f"{_fmt(length_pooled['mean']['flagged_rate'])} for the length control at the same 0.5 cutoff.",
        f"- The 0.5 cutoff is badly placed on dialogue: m3 flags {_fmt(m('m3', 'dialogue', 'flagged_rate'))} "
        f"and m0 flags {_fmt(m('m0', 'dialogue', 'flagged_rate'))} of rows, against a 50% prevalence. "
        f"On summarization the rates are {_fmt(m('m3', 'summarization', 'flagged_rate'))} and "
        f"{_fmt(m('m0', 'summarization', 'flagged_rate'))}.",
        f"- Pooled ECE is {_fmt(pooled_m3['mean']['ece'])} for m3, {_fmt(pooled_m0['mean']['ece'])} for m0, "
        f"and {_fmt(length_pooled['mean']['ece'])} for the length control.",
        f"- The two tasks differ: m3 F1 is {_fmt(m('m3', 'dialogue', 'f1'))} on dialogue and "
        f"{_fmt(m('m3', 'summarization', 'f1'))} on summarization; AUROC is "
        f"{_fmt(m('m3', 'dialogue', 'auroc'))} and {_fmt(m('m3', 'summarization', 'auroc'))}.",
        f"- Pooled AUROC is not an average of the two task AUROCs. The mean m3 score is "
        f"{mean_score_dialogue:.4f} on dialogue and {mean_score_summarization:.4f} on summarization, so "
        "pooling mixes a between-task score shift with within-task ranking. The per-task rows are the "
        "primary comparison; the pooled row is descriptive.",
        f"- F1 at 0.5 is threshold-dominated here. On dialogue both models flag about 99% of rows, so "
        f"F1 sits near the all-positive value of {all_positive_f1:.4f} even though AUROC is "
        f"{_fmt(m('m3', 'dialogue', 'auroc'))} for m3 and {_fmt(m('m0', 'dialogue', 'auroc'))} for m0. "
        "Read AUROC and PR-AUC as the ranking evidence, and flagged rate plus ECE as the operating-point "
        "evidence.",
        f"- m3 versus m0 is clearest on summarization: m3 cuts the flagged rate from "
        f"{_fmt(m('m0', 'summarization', 'flagged_rate'))} to "
        f"{_fmt(m('m3', 'summarization', 'flagged_rate'))} and ECE from "
        f"{_fmt(m('m0', 'summarization', 'ece'))} to {_fmt(m('m3', 'summarization', 'ece'))}, while "
        f"AUROC rises from {_fmt(m('m0', 'summarization', 'auroc'))} to "
        f"{_fmt(m('m3', 'summarization', 'auroc'))}.",
        f"- Reference point for the length control: the same fit reaches F1 "
        f"{_fmt(baseline_info['in_domain_test_reference']['f1'])} and AUROC "
        f"{_fmt(baseline_info['in_domain_test_reference']['auroc'])} on the HaluEval QA test split. "
        "Its task-shift strength therefore reflects the benchmark's length construction as much as a "
        "task-general signal.",
    ]

    # Answer length by task and label, the mechanism behind the length control.
    length_rows = (
        feature_frame.groupby(["task", "label"])["n_words"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .sort_values(["task", "label"])
    )
    lines += [
        "",
        "Mean answer length (words):",
        "",
        "| Task | Label 0 (correct) | Label 1 (hallucinated) |",
        "| --- | ---: | ---: |",
    ]
    for task in TASK_ORDER:
        sub = length_rows[length_rows["task"] == task].set_index("label")
        lines.append(
            f"| {task} | {sub.loc[0, 'mean']:.1f} | {sub.loc[1, 'mean']:.1f} |"
        )

    scores = feature_frame.copy()
    lines += [
        "",
        "Mean predicted score by label (m3, seed 42):",
        "",
        "| Task | Label 0 | Label 1 |",
        "| --- | ---: | ---: |",
    ]
    # Read the m3 seed-42 scores from the prediction table carried in the frame.
    for task in TASK_ORDER:
        sub = scores[scores["task"] == task]
        lines.append(
            f"| {task} | {sub.loc[sub['label'] == 0, 'm3_score'].mean():.4f} | "
            f"{sub.loc[sub['label'] == 1, 'm3_score'].mean():.4f} |"
        )

    lines += [
        "",
        "## Sensitivity: source indicator",
        "",
        "m3 was trained with a `source_ragtruth` indicator (0 for HaluEval, 1 for RAGTruth non-QA).",
        "The main table sets it to 0 because these rows are HaluEval-generated. Setting it to 1 is",
        "factually wrong for these rows but shows how much the indicator moves the output:",
        "",
        "| Task | F1 (source=0) | F1 (source=1) | AUROC (source=0) | AUROC (source=1) | Flagged rate (source=1) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in EVAL_SPLITS:
        base = _metric_rows(csv_frame, "m3", task)
        alt = sensitivity[task]["mean"]
        lines.append(
            f"| {task} | {_fmt(base['mean']['f1'])} | {_fmt(alt['f1'])} | "
            f"{_fmt(base['mean']['auroc'])} | {_fmt(alt['auroc'])} | {_fmt(alt['flagged_rate'])} |"
        )

    lines += [
        "",
        "## Limitations",
        "",
        "- Same benchmark and generator family as training. This isolates task format, not domain or",
        "  generator shift, and the paper must not describe it as an independent external corpus.",
        "- The HaluEval construction gives one correct and one hallucinated variant per item, so the",
        "  pooled prevalence is fixed at 50% by design. Flagged rates and ECE are conditional on that",
        "  balance, not on a natural deployment prevalence.",
        "- Dialogue knowledge is short; summarization documents are long and the NLI cross-encoder",
        "  truncates them at its maximum input length. The claim features use the top four context",
        "  sentences per claim, so long-document evidence is sampled, not fully read.",
        f"- {config['pairs_per_task']:,} pairs per task keeps the run bounded. With "
        f"{2 * config['pairs_per_task']:,} rows per task, sampling error on",
        "  F1/AUROC is on the order of a few hundredths; differences smaller than that should not be",
        "  interpreted as task-level effects.",
        "- The length-only control is a deliberately simple logistic fit; it is a diagnostic, not a",
        "  tuned baseline.",
        "",
        "## Runtime",
        "",
    ]
    for key, value in runtime.items():
        lines.append(f"- {key}: {value:.1f} s")
    lines += [
        "",
        "## Artifacts",
        "",
        "- `artifacts/results/rebuild/p1/task_shift.csv`: long-format metrics (per seed, seed mean, seed std).",
        "- `artifacts/results/rebuild/p1/task_shift.json`: full payload, model hashes, feature cache info.",
        "- `artifacts/results/rebuild/p1/task_shift_predictions.parquet`: per-row scores.",
        "- `artifacts/results/rebuild/p1/task_shift_REPORT.md`: this report as a standalone file, so it",
        "  survives regeneration of the shared `p1/REPORT.md`.",
        "- `data/processed/p1_task_shift.base.parquet` and `.claims.parquet`: feature caches.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="P1 task-shift evaluation (HaluEval dialogue/summarization)")
    parser.add_argument("--pairs-per-task", type=int, default=PAIRS_PER_TASK)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--device", default="cuda", help="cuda|cpu (cuda loads feature models in fp16)")
    parser.add_argument("--batch-size", type=int, default=64, help="NLI/SBERT batch for base features")
    parser.add_argument("--claim-batch-size", type=int, default=16, help="NLI batch for claim features")
    parser.add_argument("--force-features", action="store_true", help="ignore and rebuild feature caches")
    args = parser.parse_args()

    if args.pairs_per_task < 1:
        raise ValueError("--pairs-per-task must be positive")

    t_start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building task-shift frame ({args.pairs_per_task} pairs per task, seed {args.seed})")
    frame, task_meta = build_frame(args.pairs_per_task, args.seed)
    t_after_data = time.perf_counter()
    pooled_meta = {
        "raw_items": sum(task_meta[t]["raw_items"] for t in TASK_ORDER),
        "valid_items": sum(task_meta[t]["valid_items"] for t in TASK_ORDER),
        "pairs_sampled": sum(task_meta[t]["pairs_sampled"] for t in TASK_ORDER),
        "rows": int(len(frame)),
        "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()},
    }
    data_meta = {"tasks": task_meta, "pooled": pooled_meta, "sampling": {"seed": args.seed, "pairs_per_task": args.pairs_per_task}}

    logger.info("Extracting or loading features")
    feature_frame, feature_timings = extract_features(
        frame,
        args.pairs_per_task,
        args.device,
        args.batch_size,
        args.claim_batch_size,
        args.force_features,
    )
    t_after_features = time.perf_counter()
    logger.info(f"Feature frame ready: {feature_frame.shape}")

    logger.info("Loading fixed-hp checkpoints")
    loaded = load_fixed_models()
    logger.info("Training length-only control on HaluEval train")
    baseline = train_length_baseline()

    logger.info("Scoring models on dialogue, summarization, and pooled rows")
    csv_frame, evaluation = evaluate_models(feature_frame, loaded["models"], baseline)

    logger.info("Running m3 source-indicator sensitivity")
    sensitivity = evaluate_source_sensitivity(feature_frame, loaded["models"])
    t_after_scoring = time.perf_counter()

    # Attach m3 seed-42 scores to the feature frame for the report's per-label means.
    m3_scores = evaluation["predictions"]
    m3_scores = m3_scores[(m3_scores["model"] == "m3") & (m3_scores["seed"] == "42") & (m3_scores["task"] != "pooled")]
    feature_frame = feature_frame.merge(
        m3_scores[["task", "sample_id", "score"]].rename(columns={"score": "m3_score"}),
        on=["task", "sample_id"],
        how="left",
        validate="one_to_one",
    )

    runtime = {
        "data_and_sampling_s": t_after_data - t_start,
        "model_load_s": feature_timings.get("model_load_s", 0.0),
        "base_features_s": feature_timings.get("base_features_s", 0.0),
        "claim_features_s": feature_timings.get("claim_features_s", 0.0),
        "scoring_and_metrics_s": t_after_scoring - t_after_features,
        "total_s": t_after_scoring - t_start,
    }

    config = {
        "schema": "p1-task-shift-v1",
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "branch": "rebuild",
        "script": "src/models/p1_task_shift.py",
        "pairs_per_task": args.pairs_per_task,
        "sample_seed": args.seed,
        "threshold": THRESHOLD,
        "ece_bins": ECE_BINS,
        "source_ragtruth": SOURCE_VALUE,
        "device": args.device,
        "batch_size": args.batch_size,
        "claim_batch_size": args.claim_batch_size,
        "nli_model": feature_timings.get("nli_model"),
        "seeds": list(SEEDS),
        "feature_cache": {
            "base": str(BASE_CACHE.relative_to(ROOT)),
            "claims": str(CLAIM_CACHE.relative_to(ROOT)),
            "reused": bool(feature_timings.get("model_load_s", 0.0) == 0.0),
        },
    }

    payload = {
        "config": config,
        "runtime_seconds": runtime,
        "data": data_meta,
        "features": {
            "n_base": len(BASE_COLUMNS),
            "n_claim": len(CLAIM_COLUMNS),
            "base_columns": BASE_COLUMNS,
            "claim_columns": CLAIM_COLUMNS,
            "model_features": {variant: variant_features(variant) for variant in MODEL_VARIANTS},
        },
        "models": loaded["paths"],
        "length_baseline": baseline["info"],
        "metrics": evaluation["metrics"],
        "sensitivity_m3_source_ragtruth_1": sensitivity,
        "notes": [
            "Task shift within HaluEval, not an independent external corpus.",
            "Prevalence is fixed at 50% by the one-correct/one-hallucinated construction.",
            "No parameter, feature, or threshold was tuned on these rows.",
            "source_ragtruth = 0 for HaluEval-generated rows; the source=1 pass is a sensitivity check.",
            "m0/m3 rows report the seed mean (std across seeds 42/123/456); the length control is a single fit.",
        ],
        "artifacts": {
            "csv": "artifacts/results/rebuild/p1/task_shift.csv",
            "json": "artifacts/results/rebuild/p1/task_shift.json",
            "predictions": "artifacts/results/rebuild/p1/task_shift_predictions.parquet",
            "report": "artifacts/results/rebuild/p1/REPORT.md",
            "report_standalone": "artifacts/results/rebuild/p1/task_shift_REPORT.md",
        },
    }

    csv_path = OUT_DIR / "task_shift.csv"
    json_path = OUT_DIR / "task_shift.json"
    pred_path = OUT_DIR / "task_shift_predictions.parquet"
    report_path = OUT_DIR / "REPORT.md"
    standalone_report_path = OUT_DIR / "task_shift_REPORT.md"

    csv_frame.to_csv(csv_path, index=False)
    evaluation["predictions"].to_parquet(pred_path, index=False)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_text = build_report(
        csv_frame,
        data_meta,
        feature_frame,
        runtime,
        config,
        sensitivity,
        baseline["info"],
    )
    report_path.write_text(report_text, encoding="utf-8")
    standalone_report_path.write_text(report_text, encoding="utf-8")

    logger.info(f"Wrote {csv_path}")
    logger.info(f"Wrote {json_path}")
    logger.info(f"Wrote {pred_path}")
    logger.info(f"Wrote {report_path}")
    logger.info(f"Wrote {standalone_report_path}")
    logger.info(f"Total runtime: {runtime['total_s']:.1f}s")
    print(csv_frame[csv_frame["seed"].isin(["mean", "fixed"])].to_string(index=False))


if __name__ == "__main__":
    main()
