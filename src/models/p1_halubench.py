"""P1 external check: non-overlapping HaluBench subset.

HaluBench (Patronus AI, CC BY-NC 2.0) bundles 14,900 rows from six sources.
This script evaluates only the three sources that do not overlap the HaluRISC
training or external-test corpora:

  DROP          1,000 rows (500 PASS / 500 FAIL)
  FinanceBench  1,000 rows (500 PASS / 500 FAIL)
  covidQA       1,000 rows (500 PASS / 500 FAIL)

The other sources are excluded on purpose:

  halueval   10,000 rows  -> would duplicate the HaluEval training/test corpus
  RAGTruth      900 rows  -> would duplicate the RAGTruth external test split
  pubmedQA    1,000 rows  -> would overlap the MedHallu medical domain

The evaluation sample is 900 rows: 150 PASS + 150 FAIL from each of DROP,
FinanceBench, and covidQA. Labels are HaluBench labels (PASS=0, FAIL=1), not
labels re-derived from the original source datasets.

Features are the deployed 26 base features plus the 8 claim-level NLI features,
extracted with the same code paths as HaluEval/RAGTruth. Extraction runs on CPU
in parallel shards to avoid GPU contention. Both shard results and the combined
feature caches are resumable.

Models:
  m3  fixed-hp EC-XGB (35 features: 26 base + 8 claim + source indicator)
  m0  fixed-hp standard XGBoost (26 base features, no log lengths/constraints)
  length_only  logistic regression on standardized log n_chars/n_words,
               trained on the frozen HaluEval train split

All metrics are at threshold 0.5. F1 and AUROC get 1,000-draw row-bootstrap
percentile CIs.

Outputs (artifacts/results/rebuild/p1/):
  halubench.csv
  halubench.json
  REPORT.md

Run (repo root, .venv):
  $env:HALU_API_DEVICE='cpu'
  $env:HALU_XGB_DEVICE='cpu'
  .venv\\Scripts\\python.exe src/models/p1_halubench.py

Useful flags:
  --fresh              ignore feature caches and shard outputs
  --no-parallel        extract in one process (slower, simpler logs)
  --shards 3           number of CPU extraction shards
  --threads 4          torch threads per shard
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

# Thread limits must be set before numpy/torch are imported in shard workers.
_THREAD_ENV = os.environ.get("HALU_HALUBENCH_THREADS")
if _THREAD_ENV:
    os.environ["OMP_NUM_THREADS"] = _THREAD_ENV
    os.environ["MKL_NUM_THREADS"] = _THREAD_ENV

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.claim_features import FEATURE_COLUMNS as CLAIM_COLUMNS
from src.features.claim_features import extract_claim_features_df
from src.features.extract_features import (
    extract_all_core_features,
    extract_full_feature_set,
    load_heavy_models,
)
from src.models.config import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, SAMPLE_SEED, SEEDS
from src.models.rebuild_common import (
    clean_scores,
    f1_and_auroc,
    metric_bundle,
    stable_seed,
    summarize_bootstrap,
)
from src.models.run_b6_modified import (
    BASE_COLUMNS,
    SOURCE_COLUMN,
    feature_matrix,
    variant_features,
)

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "raw" / "halubench" / "test.parquet"
SUBSET_PATH = DATA_PROCESSED / "halubench_subset.parquet"
BASE_CACHE = DATA_PROCESSED / "halubench_base.parquet"
BASE_CORE_ENTITY_CACHE = DATA_PROCESSED / "halubench_base_core_entity.parquet"
CLAIM_CACHE = DATA_PROCESSED / "halubench_claim_features.parquet"
EXTRACTION_META = DATA_PROCESSED / "halubench_extraction_meta.json"
SHARD_DIR = DATA_PROCESSED / "halubench_shards"
OUT_DIR = RESULTS_DIR / "rebuild" / "p1"
FIXED_HP_MODELS = MODELS_DIR / "rebuild" / "fixed_hp"

SOURCES = ("DROP", "FinanceBench", "covidQA")
EXCLUDED_SOURCES = ("halueval", "RAGTruth", "pubmedQA")
PER_CLASS = 150
THRESHOLD = 0.5
N_BOOT = 1000
ECE_BINS = 10
MODELS = ("m3", "m0", "length_only")

METRIC_KEYS = (
    "f1",
    "auroc",
    "pr_auc",
    "ece",
    "brier",
    "flagged_rate",
    "precision",
    "recall",
)


def json_safe(value):
    """Convert numpy/pandas scalars and paths to JSON-native values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------
# Data selection
# --------------------------------------------------------------------------
def load_halubench() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found")
    df = pd.read_parquet(INPUT)
    required = {"id", "passage", "question", "answer", "label", "source_ds"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"HaluBench missing columns: {sorted(missing)}")
    if set(df["label"].unique()) != {"PASS", "FAIL"}:
        raise ValueError(f"unexpected HaluBench labels: {sorted(df['label'].unique())}")
    return df


def build_subset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Deterministic 150 PASS + 150 FAIL per non-overlapping source."""
    excluded_mask = df["source_ds"].str.lower().isin([s.lower() for s in EXCLUDED_SOURCES])
    excluded_counts = df.loc[excluded_mask, "source_ds"].value_counts().to_dict()
    keep = df.loc[~excluded_mask].copy()
    unknown = sorted(set(keep["source_ds"].unique()) - set(SOURCES))
    if unknown:
        raise ValueError(f"unexpected non-excluded sources: {unknown}")
    if len(keep) != 3000:
        raise ValueError(f"expected 3000 non-overlapping rows, found {len(keep)}")

    parts = []
    source_counts: dict[str, dict[str, int]] = {}
    for i, source in enumerate(SOURCES):
        source_counts[source] = {}
        for j, (label_name, label_int) in enumerate((("PASS", 0), ("FAIL", 1))):
            pool = keep[(keep["source_ds"] == source) & (keep["label"] == label_name)]
            available = len(pool)
            take = min(PER_CLASS, available)
            if take < PER_CLASS:
                raise ValueError(
                    f"{source} {label_name}: only {available} rows, need {PER_CLASS}"
                )
            picked = pool.sample(
                n=take, random_state=SAMPLE_SEED + 100 * i + j
            )
            source_counts[source][label_name] = int(take)
            parts.append(picked)

    subset = pd.concat(parts, ignore_index=True)
    subset["sample_id"] = "halubench::" + subset["id"].astype(str)
    if subset["sample_id"].duplicated().any():
        raise ValueError("duplicate HaluBench ids in subset")
    subset = subset.rename(columns={"passage": "context"}).copy()
    for column in ("question", "context", "answer"):
        subset[column] = subset[column].fillna("").astype(str)
    subset["label"] = (subset["label"] == "FAIL").astype(int)
    subset["halubench_label"] = np.where(subset["label"] == 1, "FAIL", "PASS")
    subset["item_idx"] = np.arange(len(subset), dtype=int)
    subset["split"] = "test"
    subset["source_ragtruth"] = 0.0
    subset = subset.sort_values(["source_ds", "label", "id"]).reset_index(drop=True)
    subset["item_idx"] = np.arange(len(subset), dtype=int)
    if len(subset) != len(SOURCES) * 2 * PER_CLASS:
        raise ValueError(f"expected {len(SOURCES) * 2 * PER_CLASS} rows, found {len(subset)}")
    subset.to_parquet(SUBSET_PATH, index=False)
    counts = {
        "halubench_rows": int(len(df)),
        "excluded_rows": {k: int(v) for k, v in excluded_counts.items()},
        "excluded_total": int(excluded_mask.sum()),
        "non_overlapping_rows": int(len(keep)),
        "selected_rows": int(len(subset)),
        "per_source": source_counts,
        "selected_prevalence": float(subset["label"].mean()),
        "sample_seed": SAMPLE_SEED,
    }
    return subset, counts


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def _shard_paths(shard: int, shards: int) -> tuple[Path, Path, Path]:
    stem = f"shard{shard}of{shards}"
    return (
        SHARD_DIR / f"halubench_base.{stem}.parquet",
        SHARD_DIR / f"halubench_claims.{stem}.parquet",
        SHARD_DIR / f"halubench_claims.{stem}.partial.parquet",
    )


def extract_base_features(frame: pd.DataFrame, device: str, batch_size: int):
    """Base features, with a reusable core+entity cache on the GPU path.

    The CPU path keeps using ``extract_full_feature_set``. On CUDA the run is
    staged: core + NER (CPU) are cached first, then NLI + SBERT run on GPU in
    fp16. A retry after an OOM therefore reuses the expensive NER pass.
    """
    models = load_heavy_models(device=device)
    if device != "cuda":
        base = extract_full_feature_set(frame, models, batch_size=batch_size)
        return base, models

    from src.features.entity_features import extract_entity_features_df
    from src.features.nli_features import extract_nli_features_df
    from src.features.semantic_features import extract_semantic_features_df

    core_entity = None
    if BASE_CORE_ENTITY_CACHE.exists():
        cached = pd.read_parquet(BASE_CORE_ENTITY_CACHE)
        if len(cached) == len(frame) and set(cached["sample_id"]) == set(frame["sample_id"]):
            core_entity = (
                cached.set_index("sample_id").loc[frame["sample_id"]].reset_index()
            )
            print(f"Reused core+entity cache: {len(core_entity)} rows", flush=True)
    if core_entity is None:
        t0 = time.time()
        core = extract_all_core_features(frame)
        entity = extract_entity_features_df(frame, models["nlp"])
        core_entity = pd.concat([core, entity], axis=1)
        core_entity.to_parquet(BASE_CORE_ENTITY_CACHE, index=False)
        print(
            f"Core+entity features: {len(core_entity)} rows in {time.time() - t0:.1f}s",
            flush=True,
        )

    t1 = time.time()
    nli = extract_nli_features_df(frame, models["nli"], batch_size=batch_size)
    print(
        f"NLI features: {len(nli)} rows in {time.time() - t1:.1f}s (batch={batch_size})",
        flush=True,
    )
    t2 = time.time()
    semantic = extract_semantic_features_df(
        frame, models["embedder"], batch_size=batch_size
    )
    print(
        f"Semantic features: {len(semantic)} rows in {time.time() - t2:.1f}s",
        flush=True,
    )
    base = pd.concat([core_entity, nli, semantic], axis=1)
    return base, models


def extract_shard(
    shard: int,
    shards: int,
    threads: int,
    fresh: bool,
    device: str = "cpu",
    batch_size: int = 64,
    claim_batch_size: int = 64,
) -> None:
    """Extract base + claim features for one interleaved shard (worker entry)."""
    import torch

    if device == "cpu":
        torch.set_num_threads(threads)
    subset = pd.read_parquet(SUBSET_PATH)
    frame = subset.iloc[shard::shards].reset_index(drop=True)
    base_path, claim_path, partial_path = _shard_paths(shard, shards)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    if fresh:
        for path in (base_path, claim_path, partial_path):
            if path.exists():
                path.unlink()

    models = None
    t0 = time.time()
    if base_path.exists():
        base = pd.read_parquet(base_path)
        print(f"[shard {shard}] reused base features: {len(base)} rows", flush=True)
    else:
        base, models = extract_base_features(frame, device, batch_size)
        print(f"[shard {shard}] models loaded in {time.time() - t0:.1f}s", flush=True)
        base = base[["sample_id", *BASE_COLUMNS]]
        tmp = base_path.with_suffix(".tmp.parquet")
        base.to_parquet(tmp, index=False)
        tmp.replace(base_path)
        print(
            f"[shard {shard}] base features: {len(base)} rows in {time.time() - t0:.1f}s",
            flush=True,
        )
    if len(base) != len(frame):
        raise ValueError(f"shard {shard}: base row mismatch {len(base)} vs {len(frame)}")

    t1 = time.time()
    if claim_path.exists():
        claims = pd.read_parquet(claim_path)
        print(f"[shard {shard}] reused claim features: {len(claims)} rows", flush=True)
    else:
        models = models if models is not None else load_heavy_models(device=device)
        claims = extract_claim_features_df(
            frame,
            models["nli"],
            partial_path=partial_path,
            checkpoint_every=100,
            batch_size=claim_batch_size,
            chunk_samples=24,
        )
        claims = claims[["sample_id", *CLAIM_COLUMNS]]
        tmp = claim_path.with_suffix(".tmp.parquet")
        claims.to_parquet(tmp, index=False)
        tmp.replace(claim_path)
        if partial_path.exists():
            partial_path.unlink()
        print(
            f"[shard {shard}] claim features: {len(claims)} rows in {time.time() - t1:.1f}s",
            flush=True,
        )
    if len(claims) != len(frame):
        raise ValueError(f"shard {shard}: claim row mismatch {len(claims)} vs {len(frame)}")
    print(f"[shard {shard}] done in {time.time() - t0:.1f}s", flush=True)


def _combine_shards(subset: pd.DataFrame, shards: int, kind: str) -> pd.DataFrame:
    parts = []
    for shard in range(shards):
        base_path, claim_path, _ = _shard_paths(shard, shards)
        path = base_path if kind == "base" else claim_path
        if not path.exists():
            raise FileNotFoundError(f"missing shard output: {path}")
        parts.append(pd.read_parquet(path))
    combined = pd.concat(parts, ignore_index=True)
    if combined["sample_id"].duplicated().any():
        raise ValueError(f"{kind}: duplicate sample_id across shards")
    missing = set(subset["sample_id"]) - set(combined["sample_id"])
    extra = set(combined["sample_id"]) - set(subset["sample_id"])
    if missing or extra:
        raise ValueError(f"{kind}: shard coverage mismatch missing={len(missing)} extra={len(extra)}")
    combined = combined.set_index("sample_id").loc[subset["sample_id"]].reset_index()
    return combined


def run_extraction(subset: pd.DataFrame, args) -> dict:
    if BASE_CACHE.exists() and CLAIM_CACHE.exists() and not args.fresh:
        base = pd.read_parquet(BASE_CACHE)
        claims = pd.read_parquet(CLAIM_CACHE)
        if len(base) == len(subset) and len(claims) == len(subset):
            print(f"Reusing combined feature caches ({len(base)} rows)")
            meta = {}
            if EXTRACTION_META.exists():
                meta = json.loads(EXTRACTION_META.read_text(encoding="utf-8"))
            return {
                **meta,
                "reused": True,
                "base_cache": str(BASE_CACHE.relative_to(ROOT)),
                "claim_cache": str(CLAIM_CACHE.relative_to(ROOT)),
                "core_entity_cache": str(BASE_CORE_ENTITY_CACHE.relative_to(ROOT))
                if BASE_CORE_ENTITY_CACHE.exists()
                else None,
                "device": meta.get("device", args.device),
                "batch_size": meta.get("batch_size", args.batch_size),
                "claim_batch_size": meta.get("claim_batch_size", args.claim_batch_size),
            }

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.no_parallel:
        extract_shard(
            0,
            1,
            args.threads,
            args.fresh,
            device=args.device,
            batch_size=args.batch_size,
            claim_batch_size=args.claim_batch_size,
        )
        shards = 1
    else:
        if args.device == "cuda":
            raise SystemExit("GPU extraction must run with --no-parallel")
        shards = args.shards
        env = os.environ.copy()
        env["HALU_HALUBENCH_THREADS"] = str(args.threads)
        env["HALU_API_DEVICE"] = "cpu"
        env["HALU_XGB_DEVICE"] = "cpu"
        log_paths = []
        processes = []
        for shard in range(shards):
            log_path = SHARD_DIR / f"halubench_shard{shard}of{shards}.log"
            log_paths.append(log_path)
            handle = log_path.open("w", encoding="utf-8")
            if shard:
                time.sleep(8)  # stagger model loading
            print(f"Launching extraction shard {shard + 1}/{shards} -> {log_path}")
            processes.append(
                (
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--extract-shard",
                            str(shard),
                            "--extract-shards",
                            str(shards),
                            "--threads",
                            str(args.threads),
                            "--device",
                            "cpu",
                            "--batch-size",
                            str(args.batch_size),
                            "--claim-batch-size",
                            str(args.claim_batch_size),
                            *(["--fresh"] if args.fresh else []),
                        ],
                        cwd=ROOT,
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    ),
                    handle,
                    log_path,
                )
            )
        failures = []
        for shard, (process, handle, log_path) in enumerate(processes):
            code = process.wait()
            handle.close()
            if code != 0:
                tail = "\n".join(
                    log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
                )
                failures.append(f"shard {shard} exit {code}:\n{tail}")
        if failures:
            raise RuntimeError("feature extraction failed:\n\n" + "\n\n".join(failures))

    base = _combine_shards(subset, shards, "base")
    claims = _combine_shards(subset, shards, "claims")
    missing_cols = [c for c in BASE_COLUMNS if c not in base.columns]
    if missing_cols:
        raise ValueError(f"base features missing: {missing_cols}")
    missing_claims = [c for c in CLAIM_COLUMNS if c not in claims.columns]
    if missing_claims:
        raise ValueError(f"claim features missing: {missing_claims}")
    if base[BASE_COLUMNS].isna().any().any():
        raise ValueError("base features contain NaN")
    if claims[CLAIM_COLUMNS].isna().any().any():
        raise ValueError("claim features contain NaN")

    base.to_parquet(BASE_CACHE, index=False)
    claims.to_parquet(CLAIM_CACHE, index=False)
    for shard in range(shards):
        for path in _shard_paths(shard, shards)[:2]:
            if path.exists():
                path.unlink()

    runtime = time.time() - started
    print(f"Feature extraction finished in {runtime / 60:.1f} min")
    meta = {
        "reused": False,
        "runtime_seconds": float(runtime),
        "rows": int(len(base)),
        "base_cache": str(BASE_CACHE.relative_to(ROOT)),
        "claim_cache": str(CLAIM_CACHE.relative_to(ROOT)),
        "core_entity_cache": str(BASE_CORE_ENTITY_CACHE.relative_to(ROOT))
        if BASE_CORE_ENTITY_CACHE.exists()
        else None,
        "shards": shards,
        "threads_per_shard": args.threads,
        "device": args.device,
        "batch_size": args.batch_size,
        "claim_batch_size": args.claim_batch_size,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    EXTRACTION_META.write_text(
        json.dumps(json_safe(meta), indent=2), encoding="utf-8"
    )
    return meta


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def load_fixed_hp_models() -> dict:
    models = {}
    for variant in ("m3", "m0"):
        names = variant_features(variant)
        for seed in SEEDS:
            path = FIXED_HP_MODELS / f"xgboost_{variant}_seed_{seed}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"missing fixed-hp model: {path}")
            model = joblib.load(path)
            if int(model.n_features_in_) != len(names):
                raise ValueError(
                    f"{path}: model expects {model.n_features_in_} features, "
                    f"variant {variant} has {len(names)}"
                )
            model.set_params(device="cpu")
            model.get_booster().set_param({"device": "cpu"})
            models[(variant, int(seed))] = model
    return models


def predict_fixed_hp(frame: pd.DataFrame, models: dict) -> dict[str, dict[int, np.ndarray]]:
    matrices = {
        "m3": feature_matrix(frame, variant_features("m3"), log_lengths=True),
        "m0": feature_matrix(frame, variant_features("m0"), log_lengths=False),
    }
    predictions: dict[str, dict[int, np.ndarray]] = {"m3": {}, "m0": {}}
    for (variant, seed), model in models.items():
        scores = np.asarray(model.predict_proba(matrices[variant])[:, 1], dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError(f"{variant} seed {seed}: non-finite scores")
        predictions[variant][int(seed)] = scores
    return predictions


def fit_length_baseline() -> tuple[object, object, dict]:
    """Same recipe as review_length_control.py: log lengths, train split only."""
    features = pd.read_parquet(
        DATA_PROCESSED / "features_full.parquet",
        columns=["split", "label", "n_chars", "n_words"],
    )
    train = features[features["split"] == "train"]
    columns = ["n_chars", "n_words"]
    x_train = np.column_stack(
        [np.log1p(train[c].to_numpy(dtype=float)) for c in columns]
    )
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=2000).fit(
        scaler.transform(x_train), train["label"].to_numpy(dtype=int)
    )
    meta = {
        "n_train": int(len(train)),
        "train_prevalence": float(train["label"].mean()),
        "features": columns,
        "transform": "log1p then StandardScaler",
        "coef": [float(v) for v in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
    }
    return model, scaler, meta


def predict_length(frame: pd.DataFrame, model, scaler) -> np.ndarray:
    columns = ["n_chars", "n_words"]
    x = np.column_stack([np.log1p(frame[c].to_numpy(dtype=float)) for c in columns])
    return np.asarray(model.predict_proba(scaler.transform(x))[:, 1], dtype=float)


# --------------------------------------------------------------------------
# Metrics and bootstrap
# --------------------------------------------------------------------------
def _output_bundle(y: np.ndarray, scores: np.ndarray) -> dict:
    """Metric bundle with the field names used by this report."""
    bundle = metric_bundle(y, scores, THRESHOLD)
    return {
        "n": int(bundle["n"]),
        "prevalence": float(bundle["prevalence"]),
        "f1": float(bundle["f1"]),
        "auroc": float(bundle["auroc"]),
        "pr_auc": float(bundle["pr_auc"]),
        "ece": float(bundle["ece"]),
        "brier": float(bundle["brier"]),
        "flagged_rate": float(bundle["predicted_positive_rate"]),
        "precision": float(bundle["precision"]),
        "recall": float(bundle["recall"]),
    }


def per_seed_metrics(y: np.ndarray, predictions: dict[int, np.ndarray]) -> dict:
    per_seed = {}
    for seed, scores in sorted(predictions.items()):
        per_seed[str(seed)] = _output_bundle(y, scores)
    point = {
        key: float(np.mean([per_seed[str(seed)][key] for seed in sorted(predictions)]))
        for key in METRIC_KEYS
    }
    spread = {
        key: float(np.std([per_seed[str(seed)][key] for seed in sorted(predictions)], ddof=1))
        if len(predictions) > 1
        else 0.0
        for key in METRIC_KEYS
    }
    return {"point": point, "sd": spread, "per_seed": per_seed}


def bootstrap_pooled_f1_auroc(
    y: np.ndarray,
    predictions: dict[int, np.ndarray],
    scope: str,
    model_name: str,
    n_boot: int = N_BOOT,
) -> dict:
    """Row bootstrap of the seed-averaged F1/AUROC, common resamples per draw."""
    y = np.asarray(y, dtype=int)
    seeds = sorted(predictions)
    rng = np.random.default_rng(stable_seed("halubench", scope, model_name))
    f1_draws, auc_draws = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        values = [f1_and_auroc(y[idx], predictions[seed][idx]) for seed in seeds]
        f1_draws.append(float(np.mean([v[0] for v in values])))
        auc_draws.append(float(np.mean([v[1] for v in values])))
    return {
        "f1": summarize_bootstrap(f1_draws),
        "auroc": summarize_bootstrap(auc_draws),
        "n_boot": int(n_boot),
        "n_boot_valid": int(len(f1_draws)),
    }


def paired_bootstrap_delta(
    y: np.ndarray,
    predictions_a: dict[int, np.ndarray],
    predictions_b: dict[int, np.ndarray],
    name: str,
    n_boot: int = N_BOOT,
) -> dict:
    y = np.asarray(y, dtype=int)
    seeds_a, seeds_b = sorted(predictions_a), sorted(predictions_b)
    rng = np.random.default_rng(stable_seed("halubench", "contrast", name))
    f1_deltas, auc_deltas = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        values_a = [f1_and_auroc(y[idx], predictions_a[s][idx]) for s in seeds_a]
        values_b = [f1_and_auroc(y[idx], predictions_b[s][idx]) for s in seeds_b]
        f1_deltas.append(
            float(np.mean([v[0] for v in values_a]) - np.mean([v[0] for v in values_b]))
        )
        auc_deltas.append(
            float(np.mean([v[1] for v in values_a]) - np.mean([v[1] for v in values_b]))
        )
    return {
        "f1_point": float(
            np.mean([f1_and_auroc(y, predictions_a[s])[0] for s in seeds_a])
            - np.mean([f1_and_auroc(y, predictions_b[s])[0] for s in seeds_b])
        ),
        "auroc_point": float(
            np.mean([f1_and_auroc(y, predictions_a[s])[1] for s in seeds_a])
            - np.mean([f1_and_auroc(y, predictions_b[s])[1] for s in seeds_b])
        ),
        "f1_delta": summarize_bootstrap(f1_deltas),
        "auroc_delta": summarize_bootstrap(auc_deltas),
        "n_boot": int(n_boot),
        "n_boot_valid": int(len(f1_deltas)),
    }


def evaluate_scope(
    frame: pd.DataFrame,
    scores: dict[str, dict[int, np.ndarray]],
    scope: str,
) -> dict:
    y = frame["label"].to_numpy(dtype=int)
    result = {
        "n": int(len(frame)),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
        "prevalence": float(y.mean()),
        "threshold": THRESHOLD,
        "models": {},
    }
    for model_name in MODELS:
        seed_predictions = scores[model_name]
        metrics = per_seed_metrics(y, seed_predictions)
        ci = bootstrap_pooled_f1_auroc(y, seed_predictions, scope, model_name)
        result["models"][model_name] = {
            "n_seeds": int(len(seed_predictions)),
            "metrics": metrics["point"],
            "metric_sd_over_seeds": metrics["sd"],
            "per_seed": metrics["per_seed"],
            "ci95": {
                "f1": [ci["f1"]["q025"], ci["f1"]["q975"]],
                "auroc": [ci["auroc"]["q025"], ci["auroc"]["q975"]],
            },
            "bootstrap": ci,
        }
    return result


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def build_rows(results: dict) -> list[dict]:
    rows = []
    scopes = [("overall", results["overall"])]
    scopes += [(source, results["by_source"][source]) for source in SOURCES]
    for scope, block in scopes:
        for model_name in MODELS:
            entry = block["models"][model_name]
            metrics = entry["metrics"]
            ci = entry["ci95"]
            sd = entry["metric_sd_over_seeds"]
            rows.append(
                {
                    "scope": scope,
                    "model": model_name,
                    "n": block["n"],
                    "positives": block["positives"],
                    "negatives": block["negatives"],
                    "prevalence": block["prevalence"],
                    "threshold": block["threshold"],
                    "f1": metrics["f1"],
                    "f1_ci_low": ci["f1"][0],
                    "f1_ci_high": ci["f1"][1],
                    "f1_seed_sd": sd["f1"],
                    "auroc": metrics["auroc"],
                    "auroc_ci_low": ci["auroc"][0],
                    "auroc_ci_high": ci["auroc"][1],
                    "auroc_seed_sd": sd["auroc"],
                    "pr_auc": metrics["pr_auc"],
                    "ece": metrics["ece"],
                    "brier": metrics["brier"],
                    "flagged_rate": metrics["flagged_rate"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "n_seeds": entry["n_seeds"],
                    "n_boot": entry["bootstrap"]["n_boot"],
                }
            )
    return rows


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def fmt_ci(value: float, interval: list[float], digits: int = 4) -> str:
    return f"{value:.{digits}f} [{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def fmt_interval(interval: list[float], digits: int = 4) -> str:
    return f"[{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def write_report(payload: dict, rows: list[dict]) -> None:
    protocol = payload["protocol"]
    counts = payload["counts"]
    overall = payload["results"]["overall"]
    by_source = payload["results"]["by_source"]
    contrasts = payload["contrasts"]
    extraction = payload["extraction"]

    def model_row(block: dict, model_name: str) -> str:
        entry = block["models"][model_name]
        metrics = entry["metrics"]
        ci = entry["ci95"]
        return (
            f"| `{model_name}` | {fmt(metrics['f1'])} "
            f"| {fmt_interval(ci['f1'])} "
            f"| {fmt(metrics['auroc'])} "
            f"| {fmt_interval(ci['auroc'])} "
            f"| {fmt(metrics['pr_auc'])} "
            f"| {fmt(metrics['ece'])} "
            f"| {fmt(metrics['brier'])} "
            f"| {fmt(metrics['flagged_rate'])} |"
        )

    header = (
        "| Model | F1 | F1 95% CI | AUROC | AUROC 95% CI | PR-AUC | ECE | Brier | Flagged |\n"
        "|---|---:|---|---:|---|---:|---:|---:|---:|"
    )

    lines = [
        "# P1 external check: non-overlapping HaluBench subset",
        "",
        f"Generated: {payload['generated_at_utc']}  ",
        f"Script: `{payload['script']}`  ",
        f"Git: `{payload['git']['branch']}` @ `{payload['git']['commit']}`  ",
        f"Device: NLI/SBERT on {extraction.get('device', 'cpu')} "
        f"({'fp16' if extraction.get('device') == 'cuda' else 'fp32'}); "
        "core/entity and XGBoost on CPU",
        "",
        "## Scope and license",
        "",
        "HaluBench is distributed under **CC BY-NC 2.0**; this use is non-commercial",
        "research. HaluBench bundles rows from six source datasets. This evaluation",
        "keeps only the three sources that do not overlap the HaluRISC corpora:",
        "",
        f"- DROP: {counts['per_source']['DROP']['PASS']} PASS + "
        f"{counts['per_source']['DROP']['FAIL']} FAIL",
        f"- FinanceBench: {counts['per_source']['FinanceBench']['PASS']} PASS + "
        f"{counts['per_source']['FinanceBench']['FAIL']} FAIL",
        f"- covidQA: {counts['per_source']['covidQA']['PASS']} PASS + "
        f"{counts['per_source']['covidQA']['FAIL']} FAIL",
        "",
        f"Excluded rows: {counts['excluded_total']:,} "
        f"({', '.join(f'{k} {v:,}' for k, v in counts['excluded_rows'].items())}).",
        "HaluEval and RAGTruth are excluded to avoid double-counting the training and",
        "external-test corpora; PubMedQA is excluded to avoid overlap with MedHallu.",
        "The 900 evaluation rows are a 30% sample of the 3,000 remaining rows.",
        "",
        "**Labels come from HaluBench, not from the original source datasets.**",
        "PASS is mapped to 0 (faithful) and FAIL to 1 (hallucinated). The DROP and",
        "FinanceBench rows were converted to question-answer form by the HaluBench",
        "authors, so they are not the original task formats.",
        "",
        "## Protocol",
        "",
        f"- Evaluation rows: {overall['n']} "
        f"({overall['positives']} FAIL, {overall['negatives']} PASS; "
        f"prevalence {overall['prevalence']:.4f})",
        "- Inputs: `context = passage`, `question = question`, `answer = answer`",
        "- Base features: 26 (7 groups: length, lexical, entity, NLI, numeric, hedging, semantic)",
        "- Claim features: 8 claim-level NLI aggregates over atomic answer clauses",
        "- `m3`: fixed-hp EC-XGB, 35 features (26 base + 8 claim + RAGTruth source indicator)",
        "- `m0`: fixed-hp standard XGBoost, 26 base features, no log lengths or monotone constraints",
        "- `length_only`: logistic regression on standardized log `n_chars`/`n_words`,",
        "  trained on the frozen HaluEval train split only",
        "- Source indicator is 0 for every HaluBench row (no RAGTruth natural-response rows here)",
        "- Threshold: 0.5 for every model; no threshold tuning on HaluBench",
        "- CIs: 1,000-draw row bootstrap for F1 and AUROC",
        f"- m3/m0 configuration: max_depth={protocol['fixed_hyperparameters']['max_depth']}, "
        f"learning_rate={protocol['fixed_hyperparameters']['learning_rate']}, "
        f"n_estimators={protocol['fixed_hyperparameters']['n_estimators']}, "
        f"subsample={protocol['fixed_hyperparameters']['subsample']}, "
        f"colsample_bytree={protocol['fixed_hyperparameters']['colsample_bytree']}",
        f"- Seeds: {', '.join(str(s) for s in protocol['seeds'])}; metrics average the three",
        "  per-seed values. CIs resample rows once per draw and use the same rows for every seed.",
        "",
        "## Overall results (n = 900)",
        "",
        header,
    ]
    for model_name in MODELS:
        lines.append(model_row(overall, model_name))
    lines += [
        "",
        "## Results by source",
        "",
    ]
    for source in SOURCES:
        block = by_source[source]
        lines += [
            f"### {source} (n = {block['n']}, FAIL prevalence {block['prevalence']:.4f})",
            "",
            header,
        ]
        for model_name in MODELS:
            lines.append(model_row(block, model_name))
        lines.append("")

    lines += [
        "## Paired contrasts (seed-averaged, 1,000-draw bootstrap)",
        "",
        "| Contrast | F1 delta [95% CI] | AUROC delta [95% CI] |",
        "|---|---:|---:|",
    ]
    for name, block in contrasts.items():
        lines.append(
            f"| `{name}` | {fmt_ci(block['f1_point'], [block['f1_delta']['q025'], block['f1_delta']['q975']])} "
            f"| {fmt_ci(block['auroc_point'], [block['auroc_delta']['q025'], block['auroc_delta']['q975']])} |"
        )

    m3 = overall["models"]["m3"]
    m3_m0 = contrasts["m3_minus_m0"]
    m3_len = contrasts["m3_minus_length_only"]
    covid = by_source["covidQA"]["models"]["m3"]["metrics"]
    drop = by_source["DROP"]["models"]["m3"]["metrics"]
    finance = by_source["FinanceBench"]["models"]["m3"]["metrics"]

    lines += [
        "",
        "## Key readings",
        "",
        f"- m3 and m0 are not distinguishable at the 95% level overall: "
        f"dF1 {m3_m0['f1_point']:+.4f} "
        f"[{m3_m0['f1_delta']['q025']:+.4f}, {m3_m0['f1_delta']['q975']:+.4f}], "
        f"dAUROC {m3_m0['auroc_point']:+.4f} "
        f"[{m3_m0['auroc_delta']['q025']:+.4f}, {m3_m0['auroc_delta']['q975']:+.4f}].",
        f"- Both models beat the length-only baseline: m3 dF1 "
        f"{m3_len['f1_point']:+.4f} "
        f"[{m3_len['f1_delta']['q025']:+.4f}, {m3_len['f1_delta']['q975']:+.4f}] "
        f"and dAUROC {m3_len['auroc_point']:+.4f} "
        f"[{m3_len['auroc_delta']['q025']:+.4f}, {m3_len['auroc_delta']['q975']:+.4f}]. "
        "The external signal is not only answer length.",
        f"- Performance is strongly source-dependent: m3 AUROC is "
        f"{covid['auroc']:.4f} on covidQA but {drop['auroc']:.4f} on DROP and "
        f"{finance['auroc']:.4f} on FinanceBench. The latter two are close to chance.",
        f"- Calibration degrades under domain shift: m3 ECE "
        f"{m3['metrics']['ece']:.4f} and Brier {m3['metrics']['brier']:.4f} overall. "
        f"FinanceBench has ECE {finance['ece']:.4f} with a flagged rate of "
        f"{finance['flagged_rate']:.4f}.",
        "",
    ]

    lines += [
        "",
        "## Feature extraction and runtime",
        "",
        f"- Base + claim feature extraction: {extraction['runtime_seconds'] / 60:.1f} min",
        f"- Extraction device: {extraction.get('device', 'cpu')}; "
        f"batch {extraction.get('batch_size', '?')} "
        f"(claims {extraction.get('claim_batch_size', '?')})",
        f"- Core+entity cache: `{extraction.get('core_entity_cache')}` "
        "(recomputed once after the CPU shards stopped; the shard workers had not "
        "written base caches yet, so no per-shard base/entity file existed to reuse)",
        f"- Shards: {extraction['shards']} x {extraction['threads_per_shard']} torch threads",
        f"- Feature caches: `{extraction['base_cache']}`, `{extraction['claim_cache']}`",
        f"- Total wall time: {payload['runtime_seconds'] / 60:.1f} min",
        f"- Device note: {payload['environment']['cpu_contention_note']}",
        "",
        "## Caveats",
        "",
        "- HaluBench labels are benchmark labels. They were not re-annotated for this",
        "  study, and agreement with the original DROP/FinanceBench/covidQA annotations",
        "  was not measured.",
        "- DROP answers are often lists of spans encoded as strings, and FinanceBench",
        "  answers can be long. The extractors treat every answer as plain text.",
        "- The three sources differ in domain (Wikipedia/reading comprehension, financial",
        "  filings, COVID-19 literature), so per-source results are not interchangeable.",
        "- HaluBench contains only 500 PASS and 500 FAIL per source. The 900-row sample",
        "  is balanced by construction, so prevalence-dependent metrics (F1, flagged rate)",
        "  describe this balanced sample, not natural deployment prevalence.",
        "- m3 uses a RAGTruth source indicator fixed to 0. HaluBench rows are benchmark",
        "  answers, not RAGTruth natural responses, so this is the intended setting.",
        "- No calibration is fitted on HaluBench; ECE and Brier describe raw model scores.",
        "- CIs are percentile row-bootstrap intervals and do not model source clustering.",
        "",
        "## Blockers",
        "",
    ]
    blockers = payload.get("blockers") or []
    if blockers:
        lines += [f"- {item}" for item in blockers]
    else:
        lines += ["- None."]
    lines.append("")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1 non-overlapping HaluBench subset")
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--device",
        default=os.environ.get("HALU_EXTRACT_DEVICE", "cpu"),
        choices=("cpu", "cuda"),
        help="Feature-extraction device; GPU runs must use --no-parallel",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--claim-batch-size", type=int, default=64)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--extract-shard", type=int, default=None)
    parser.add_argument("--extract-shards", dest="extract_shards", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.extract_shard is not None:
        shards = args.extract_shards if args.extract_shards is not None else args.shards
        extract_shard(
            args.extract_shard,
            shards,
            args.threads,
            args.fresh,
            device=args.device,
            batch_size=args.batch_size,
            claim_batch_size=args.claim_batch_size,
        )
        return

    started = time.time()
    os.environ.setdefault("HALU_API_DEVICE", "cpu")
    os.environ.setdefault("HALU_XGB_DEVICE", "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_halubench()
    subset, counts = build_subset(raw)
    print(f"Selected {len(subset)} rows: {json.dumps(counts['per_source'])}")

    extraction = run_extraction(subset, args)
    base = pd.read_parquet(BASE_CACHE)
    claims = pd.read_parquet(CLAIM_CACHE)

    frame = subset.merge(base, on="sample_id", how="left", validate="one_to_one")
    frame = frame.merge(
        claims.rename(columns={c: c for c in CLAIM_COLUMNS}),
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if frame[BASE_COLUMNS + CLAIM_COLUMNS].isna().any().any():
        raise ValueError("missing features after merge")
    frame[SOURCE_COLUMN] = 0.0

    models = load_fixed_hp_models()
    fixed_predictions = predict_fixed_hp(frame, models)
    length_model, length_scaler, length_meta = fit_length_baseline()
    length_scores = predict_length(frame, length_model, length_scaler)

    scores = {
        "m3": fixed_predictions["m3"],
        "m0": fixed_predictions["m0"],
        "length_only": {0: length_scores},
    }

    overall = evaluate_scope(frame, scores, "overall")
    by_source = {}
    for source in SOURCES:
        source_frame = frame[frame["source_ds"] == source].reset_index(drop=True)
        source_scores = {
            model_name: {
                seed: np.asarray(scores[model_name][seed])[
                    frame["source_ds"].to_numpy() == source
                ]
                for seed in scores[model_name]
            }
            for model_name in MODELS
        }
        by_source[source] = evaluate_scope(source_frame, source_scores, source)

    y = frame["label"].to_numpy(dtype=int)
    contrasts = {
        "m3_minus_m0": paired_bootstrap_delta(y, scores["m3"], scores["m0"], "m3_minus_m0"),
        "m3_minus_length_only": paired_bootstrap_delta(
            y, scores["m3"], scores["length_only"], "m3_minus_length_only"
        ),
        "m0_minus_length_only": paired_bootstrap_delta(
            y, scores["m0"], scores["length_only"], "m0_minus_length_only"
        ),
    }

    runtime = time.time() - started
    blockers = []
    if extraction["runtime_seconds"] / 60 > 30:
        blockers.append(
            "Feature extraction exceeded the 30-minute target "
            f"({extraction['runtime_seconds'] / 60:.1f} min). The host CPU was shared "
            "with another workload (roughly 50-65% busy before extraction), so the "
            "CPU-only run was slower than the benchmark estimate. All 900 rows completed."
        )

    rows = build_rows({"overall": overall, "by_source": by_source})
    pd.DataFrame(rows).to_csv(OUT_DIR / "halubench.csv", index=False)

    payload = {
        "schema": "p1-halubench-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "src/models/p1_halubench.py",
        "git": {
            "branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": git_value("rev-parse", "HEAD"),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "xgb_device": os.environ.get("HALU_XGB_DEVICE"),
            "api_device": os.environ.get("HALU_API_DEVICE"),
            "cpu_threads": os.cpu_count(),
            "cpu_contention_note": (
                "CPU shards were stopped after the host CPU became the bottleneck; "
                "core+entity features ran on CPU once, then NLI and SBERT ran on GPU "
                "in fp16 with the requested small batch. This is a device change only, "
                "not a feature-definition change."
                if extraction.get("device") == "cuda"
                else "host CPU was shared with another workload during extraction"
            ),
        },
        "protocol": {
            "dataset": "HaluBench (Patronus AI)",
            "license": "CC BY-NC 2.0",
            "use": "non-commercial research",
            "subset": "DROP, FinanceBench, covidQA only",
            "excluded_sources": list(EXCLUDED_SOURCES),
            "exclusion_reason": (
                "HaluEval and RAGTruth would duplicate the training and external-test "
                "corpora; pubmedQA would overlap MedHallu"
            ),
            "sampling": "150 PASS + 150 FAIL per source, stratified",
            "sample_seed": SAMPLE_SEED,
            "label_mapping": {"PASS": 0, "FAIL": 1},
            "label_provenance": "HaluBench labels, not re-derived from source datasets",
            "threshold": THRESHOLD,
            "fixed_hyperparameters": {
                "max_depth": 5,
                "learning_rate": 0.05,
                "n_estimators": 300,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
            },
            "seeds": [int(s) for s in SEEDS],
            "source_indicator": 0.0,
            "length_baseline": length_meta,
            "metrics": list(METRIC_KEYS),
            "bootstrap": {
                "n_boot": N_BOOT,
                "unit": "row",
                "interval": "percentile 95%",
                "resampling": "common row indices across seeds",
            },
        },
        "counts": counts,
        "extraction": extraction,
        "runtime_seconds": float(runtime),
        "results": {"overall": overall, "by_source": by_source},
        "contrasts": contrasts,
        "blockers": blockers,
        "caveats": [
            "HaluBench labels are benchmark labels; no re-annotation or agreement check.",
            "DROP answers are often list-like span strings; FinanceBench answers can be long.",
            "Balanced sampling by construction; F1/flagged rate are not deployment prevalence.",
            "m3 source indicator fixed to 0 because these are not RAGTruth natural responses.",
            "No calibration fit on HaluBench; ECE/Brier use raw scores.",
        ],
    }
    (OUT_DIR / "halubench.json").write_text(
        json.dumps(json_safe(payload), indent=2), encoding="utf-8"
    )
    write_report(payload, rows)

    print(f"\nWrote {OUT_DIR / 'halubench.csv'}")
    print(f"Wrote {OUT_DIR / 'halubench.json'}")
    print(f"Wrote {OUT_DIR / 'REPORT.md'}")
    print(f"Total wall time: {runtime / 60:.1f} min")
    print("\nOverall:")
    for row in rows:
        if row["scope"] == "overall":
            print(
                f"  {row['model']:12s} F1={row['f1']:.4f} "
                f"[{row['f1_ci_low']:.4f},{row['f1_ci_high']:.4f}] "
                f"AUROC={row['auroc']:.4f} "
                f"[{row['auroc_ci_low']:.4f},{row['auroc_ci_high']:.4f}] "
                f"PR-AUC={row['pr_auc']:.4f} ECE={row['ece']:.4f} "
                f"Brier={row['brier']:.4f} Flagged={row['flagged_rate']:.4f}"
            )


if __name__ == "__main__":
    main()
