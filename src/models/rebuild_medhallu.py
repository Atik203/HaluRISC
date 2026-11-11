"""P1 second-corpus evaluation on MedHallu (medical QA).

MedHallu (UTAustin-AIHealth/MedHallu, config ``pqa_labeled``, MIT licence) ships
1,000 human-annotated PubMedQA-style triples: a question, a set of knowledge
passages, an expert-written ground-truth answer, and an LLM-written
hallucinated answer.  This script turns each triple into one grounded row
(label 0) and one hallucinated row (label 1), extracts the same 26 base and 8
claim-level features used by the HaluEval pipeline, and evaluates the frozen
fixed-hyperparameter m0 and m3 models plus a length-only logistic baseline.

MedHallu is a different domain (biomedical QA), independently annotated, and
was not used for training or tuning.  It is reported as a second corpus, not as
a task-shift subset of HaluEval.

Run (repo root, .venv, CPU to avoid GPU contention)::

    $env:HALU_API_DEVICE="cpu"; $env:HALU_XGB_DEVICE="cpu"
    .venv\\Scripts\\python.exe src\\models\\rebuild_medhallu.py

Artifacts land in ``artifacts/results/rebuild/p1``.  Feature caches and the
2,000-row record table land in ``data/processed`` so a rerun is cheap.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.claim_features import FEATURE_COLUMNS as CLAIM_COLUMNS  # noqa: E402
from src.features.claim_features import extract_claim_features_df  # noqa: E402
from src.features.extract_features import extract_full_feature_set, load_heavy_models  # noqa: E402
from src.models.config import BOOTSTRAP_SEED, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, SEEDS  # noqa: E402
from src.models.rebuild_common import (  # noqa: E402
    METRIC_KEYS,
    clean_scores,
    f1_and_auroc,
    metric_bundle,
    m3_feature_names,
    summarize_bootstrap,
)
from src.models.run_b6_modified import (  # noqa: E402
    BASE_COLUMNS,
    SOURCE_COLUMN,
    feature_matrix,
    variant_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rebuild_medhallu")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "medhallu" / "pqa_labeled.parquet"
FIXED_HP_MODELS = MODELS_DIR / "rebuild" / "fixed_hp"
FIXED_HP_PRED = RESULTS_DIR / "rebuild" / "fixed_hp" / "fixed_hp_predictions.parquet"
FEATURES_FULL = DATA_PROCESSED / "features_full.parquet"
OUT_DIR = RESULTS_DIR / "rebuild" / "p1"

DATASET_LABEL = "MedHallu pqa_labeled"
N_PAIRS = 1000
THRESHOLD = 0.5
N_BOOT = 1000
N_FEATURES = {"m0": len(BASE_COLUMNS), "m3": len(m3_feature_names())}
PAIR_SEED = BOOTSTRAP_SEED  # 777


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
def build_records(suffix: str = "", limit: int | None = None) -> pd.DataFrame:
    """One grounded + one hallucinated row per MedHallu triple."""
    if not RAW.exists():
        raise FileNotFoundError(f"MedHallu parquet not found: {RAW}")
    raw = pd.read_parquet(RAW).reset_index(drop=True)
    required = {
        "Question",
        "Knowledge",
        "Ground Truth",
        "Hallucinated Answer",
        "Difficulty Level",
        "Category of Hallucination",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"MedHallu parquet missing columns: {sorted(missing)}")
    if limit:
        raw = raw.head(limit)

    rows = []
    for i, rec in raw.iterrows():
        knowledge = rec["Knowledge"]
        passages = [str(p).strip() for p in knowledge] if knowledge is not None else []
        passages = [p for p in passages if p]
        if not passages:
            raise ValueError(f"MedHallu row {i} has no knowledge passages")
        context = "\n\n".join(passages)
        question = str(rec["Question"]).strip()
        pair_id = f"medhallu-{i:04d}"
        base = {
            "pair_id": pair_id,
            "question": question,
            "context": context,
            "difficulty": str(rec["Difficulty Level"]).strip().lower(),
            "category": str(rec["Category of Hallucination"]).strip(),
            "n_knowledge_passages": len(passages),
        }
        for role, answer_col, label in (
            ("grounded", "Ground Truth", 0),
            ("hallucinated", "Hallucinated Answer", 1),
        ):
            answer = str(rec[answer_col]).strip()
            if not answer:
                raise ValueError(f"MedHallu row {i} has an empty {answer_col}")
            rows.append(
                {
                    "sample_id": f"{pair_id}-{role}",
                    "item_idx": i,
                    "label": label,
                    "split": "eval",
                    "role": role,
                    "answer": answer,
                    **base,
                }
            )
    out = pd.DataFrame(rows)
    out = out.sort_values(["item_idx", "label"], ascending=[True, False]).reset_index(drop=True)
    records_path = DATA_PROCESSED / f"medhallu_records{suffix}.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(records_path, index=False)
    logger.info(
        "Records: %d rows from %d pairs (labels %s) -> %s",
        len(out),
        out["pair_id"].nunique(),
        out["label"].value_counts().sort_index().to_dict(),
        records_path,
    )
    return out


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def extract_features(
    records: pd.DataFrame,
    suffix: str = "",
    base_cache: Path | None = None,
    claim_cache: Path | None = None,
    device: str = "cpu",
    base_batch_size: int = 64,
    claim_batch_size: int = 32,
) -> pd.DataFrame:
    """Extract 26 base + 8 claim features, reusing caches when complete."""
    base_cache = base_cache or DATA_PROCESSED / f"medhallu_base_features{suffix}.parquet"
    claim_cache = claim_cache or DATA_PROCESSED / f"medhallu_claim_features{suffix}.parquet"
    ids = set(records["sample_id"])

    need_base = True
    if base_cache.exists():
        cached = pd.read_parquet(base_cache)
        if ids.issubset(set(cached["sample_id"])) and set(BASE_COLUMNS).issubset(cached.columns):
            base = cached[cached["sample_id"].isin(ids)].copy()
            logger.info("Reusing base feature cache: %d rows (%s)", len(base), base_cache)
            need_base = False
        else:
            logger.info("Base feature cache incomplete; recomputing (%s)", base_cache)
    if need_base:
        models = load_heavy_models(device=device)
        logger.info("Heavy models loaded on %s (NLI=%s)", device, models.get("nli_name"))
        t0 = time.time()
        base = extract_full_feature_set(records, models, batch_size=base_batch_size)
        base.to_parquet(base_cache, index=False)
        logger.info("Base features: %s in %.1fs -> %s", base.shape, time.time() - t0, base_cache)
        # Reuse the already-loaded NLI encoder for the claim pass.
        return _claims_stage(records, base, claim_cache, models["nli"], claim_batch_size, device)

    models = load_heavy_models(device=device)
    logger.info("Heavy models loaded on %s (NLI=%s)", device, models.get("nli_name"))
    return _claims_stage(records, base, claim_cache, models["nli"], claim_batch_size, device)


def _claims_stage(records, base, claim_cache, nli_model, batch_size, device):
    ids = set(records["sample_id"])
    need_claims = True
    if claim_cache.exists():
        cached = pd.read_parquet(claim_cache)
        if ids.issubset(set(cached["sample_id"])) and set(CLAIM_COLUMNS).issubset(cached.columns):
            claims = cached[cached["sample_id"].isin(ids)].copy()
            logger.info("Reusing claim feature cache: %d rows (%s)", len(claims), claim_cache)
            need_claims = False
    if need_claims:
        partial = claim_cache.with_suffix(".partial.parquet")
        if partial.exists():
            partial.unlink()
            logger.info("Removed stale claim checkpoint: %s", partial)
        t0 = time.time()
        claims = extract_claim_features_df(
            records,
            nli_model,
            partial_path=partial,
            checkpoint_every=500,
            batch_size=batch_size,
            chunk_samples=24,
        )
        if partial.exists():
            partial.unlink()
        claims.to_parquet(claim_cache, index=False)
        logger.info("Claim features: %s in %.1fs -> %s", claims.shape, time.time() - t0, claim_cache)

    df = base.merge(claims, on="sample_id", how="inner", validate="one_to_one")
    meta = records[
        ["sample_id", "pair_id", "role", "difficulty", "category", "n_knowledge_passages"]
    ]
    df = df.merge(meta, on="sample_id", how="inner", validate="one_to_one")
    missing = [c for c in BASE_COLUMNS + CLAIM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"feature columns missing after merge: {missing}")
    if df[BASE_COLUMNS + CLAIM_COLUMNS].isna().any().any():
        bad = df[BASE_COLUMNS + CLAIM_COLUMNS].isna().sum()
        raise ValueError(f"NaN features after merge: {bad[bad > 0].to_dict()}")
    if len(df) != len(records):
        raise ValueError(f"feature merge changed row count: {len(df)} != {len(records)}")
    df[SOURCE_COLUMN] = 1.0
    df["split"] = "eval"
    df["item_idx"] = df["item_idx"].astype(int)
    df["label"] = df["label"].astype(int)
    df = df.sort_values(["item_idx", "label"], ascending=[True, False]).reset_index(drop=True)
    logger.info("Evaluation frame: %d rows x %d columns", *df.shape)
    return df


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def pair_accuracy(scores: np.ndarray, gt_idx: np.ndarray, hal_idx: np.ndarray) -> float:
    """Share of pairs where the hallucinated answer scores above the grounded one."""
    if len(gt_idx) == 0:
        return float("nan")
    diff = np.asarray(scores)[hal_idx] - np.asarray(scores)[gt_idx]
    return float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0))


def score_frame(df: pd.DataFrame, scores: np.ndarray, threshold: float = THRESHOLD) -> dict:
    """Point metrics plus the pair-level concordance."""
    y = df["label"].to_numpy(dtype=int)
    bundle = metric_bundle(y, scores, threshold)
    unique_pairs = pd.unique(df["pair_id"])
    lookup = df[["pair_id", "role"]].copy()
    lookup["row"] = np.arange(len(df))
    gt_idx = (
        lookup[lookup["role"] == "grounded"].set_index("pair_id")["row"].reindex(unique_pairs).to_numpy()
    )
    hal_idx = (
        lookup[lookup["role"] == "hallucinated"].set_index("pair_id")["row"].reindex(unique_pairs).to_numpy()
    )
    bundle["pair_accuracy"] = pair_accuracy(scores, gt_idx, hal_idx)
    bundle["mean_score"] = float(np.mean(clean_scores(scores)))
    bundle["n_pairs"] = int(len(unique_pairs))
    return bundle


def pair_bootstrap(
    df: pd.DataFrame,
    scores: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = PAIR_SEED,
    threshold: float = THRESHOLD,
) -> dict:
    """Pair-clustered bootstrap CIs for F1, AUROC and pair concordance.

    Both rows of a drawn pair enter together, so the dependence between the
    grounded and hallucinated answer is preserved.  This is the correct unit
    for a paired design; an i.i.d. row bootstrap would understate the CI.
    """
    y = df["label"].to_numpy(dtype=int)
    scores = clean_scores(scores)
    pair_ids = df["pair_id"].to_numpy()
    unique_pairs = pd.unique(pair_ids)
    lookup = df[["pair_id", "role"]].copy()
    lookup["row"] = np.arange(len(df))
    gt_idx = (
        lookup[lookup["role"] == "grounded"].set_index("pair_id")["row"].reindex(unique_pairs).to_numpy()
    )
    hal_idx = (
        lookup[lookup["role"] == "hallucinated"].set_index("pair_id")["row"].reindex(unique_pairs).to_numpy()
    )
    rows_by_pair = [np.flatnonzero(pair_ids == pid) for pid in unique_pairs]

    rng = np.random.default_rng(seed)
    draws = {"f1": [], "auroc": [], "pair_accuracy": []}
    for _ in range(n_boot):
        chosen = rng.integers(0, len(unique_pairs), len(unique_pairs))
        idx = np.concatenate([rows_by_pair[j] for j in chosen])
        y_boot = y[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        f1, auroc = f1_and_auroc(y_boot, scores[idx], threshold)
        diff = scores[hal_idx[chosen]] - scores[gt_idx[chosen]]
        draws["f1"].append(f1)
        draws["auroc"].append(auroc)
        draws["pair_accuracy"].append(float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0)))
    return {key: summarize_bootstrap(values) for key, values in draws.items()}


def metric_row(bundle: dict, ci: dict | None = None) -> dict:
    row = {
        "n": bundle["n"],
        "n_pairs": bundle.get("n_pairs", bundle["n"]),
        "prevalence": bundle["prevalence"],
        "threshold": bundle["threshold"],
        "f1": bundle["f1"],
        "auroc": bundle["auroc"],
        "pr_auc": bundle["pr_auc"],
        "ece": bundle["ece"],
        "brier": bundle["brier"],
        "flagged_rate": bundle["predicted_positive_rate"],
        "precision": bundle["precision"],
        "recall": bundle["recall"],
        "pair_accuracy": bundle.get("pair_accuracy"),
        "mean_score": bundle.get("mean_score"),
    }
    for key in ("f1", "auroc", "pair_accuracy"):
        if ci and key in ci:
            row[f"{key}_ci_low"] = ci[key]["q025"]
            row[f"{key}_ci_high"] = ci[key]["q975"]
    return row


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def load_fixed_hp_model(variant: str, seed: int):
    import joblib

    path = FIXED_HP_MODELS / f"xgboost_{variant}_seed_{seed}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"fixed-hp model not found: {path}")
    return joblib.load(path)


def predict_variant(df: pd.DataFrame, variant: str, seeds=SEEDS) -> dict[int, np.ndarray]:
    names = variant_features(variant)
    log_lengths = variant != "m0"
    X = feature_matrix(df, names, log_lengths)
    out = {}
    for seed in seeds:
        model = load_fixed_hp_model(variant, seed)
        out[int(seed)] = model.predict_proba(X)[:, 1]
    return out


def length_only_baseline(target: pd.DataFrame, seed: int = SEEDS[0]) -> tuple[np.ndarray, dict]:
    """Logistic regression on log word count, fit on HaluEval train only."""
    from sklearn.linear_model import LogisticRegression

    if not FEATURES_FULL.exists():
        raise FileNotFoundError(f"missing HaluEval features: {FEATURES_FULL}")
    train = pd.read_parquet(FEATURES_FULL, columns=["split", "label", "n_words"])
    train = train[train["split"] == "train"]
    if len(train) != 14000:
        raise ValueError(f"expected 14,000 HaluEval train rows, found {len(train)}")
    x_train = np.log1p(train["n_words"].to_numpy(dtype=float)).reshape(-1, 1)
    model = LogisticRegression(max_iter=2000, random_state=seed)
    model.fit(x_train, train["label"].to_numpy(dtype=int))
    x_target = np.log1p(target["n_words"].to_numpy(dtype=float)).reshape(-1, 1)
    scores = model.predict_proba(x_target)[:, 1]
    info = {
        "features": ["log1p(n_words)"],
        "coef": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "train_rows": int(len(train)),
        "train_prevalence": float(train["label"].mean()),
        "train_mean_words": {
            str(int(label)): float(train.loc[train["label"] == label, "n_words"].mean())
            for label in sorted(train["label"].unique())
        },
    }
    return scores, info


# --------------------------------------------------------------------------
# HaluEval reference
# --------------------------------------------------------------------------
def halueval_reference() -> dict:
    """Same fixed-hp models on the HaluEval test split (in-domain reference)."""
    if not FIXED_HP_PRED.exists():
        logger.warning("HaluEval reference predictions missing: %s", FIXED_HP_PRED)
        return {}
    pred = pd.read_parquet(FIXED_HP_PRED)
    ref = {}
    for variant in ("m0", "m3"):
        sub = pred[(pred["variant"] == variant) & (pred["dataset"] == "halueval_test")]
        per_seed = []
        for seed in SEEDS:
            s = sub[sub["seed"] == seed]
            if s.empty:
                continue
            bundle = metric_bundle(s["label"].to_numpy(), s["score"].to_numpy(), THRESHOLD)
            per_seed.append({"seed": int(seed), **metric_row(bundle)})
        if not per_seed:
            continue
        mean = {}
        for key in ("f1", "auroc", "pr_auc", "ece", "brier", "flagged_rate", "precision", "recall"):
            values = [r[key] for r in per_seed]
            mean[key] = float(np.mean(values))
            mean[f"{key}_std"] = float(np.std(values, ddof=1))
        ref[variant] = {"per_seed": per_seed, "mean": mean, "n": per_seed[0]["n"]}
    return ref


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="MedHallu second-corpus evaluation")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N pairs")
    parser.add_argument("--suffix", default="", help="suffix for processed cache files (smoke runs)")
    parser.add_argument("--out-suffix", default="", help="suffix for output files (smoke runs)")
    parser.add_argument("--device", default=os.environ.get("HALU_API_DEVICE", "cpu"))
    parser.add_argument("--base-batch-size", type=int, default=64)
    parser.add_argument("--claim-batch-size", type=int, default=32)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--skip-halueval-reference", action="store_true")
    parser.add_argument("--stage", choices=("all", "features", "eval"), default="all")
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("HALU_TORCH_THREADS", "16")),
        help="torch CPU threads for NLI/SBERT (default 16)",
    )
    args = parser.parse_args()

    os.environ.setdefault("HALU_XGB_DEVICE", "cpu")
    os.environ["HALU_API_DEVICE"] = args.device
    os.environ.setdefault("HALU_SPACY_N_PROCESS", "1")
    if args.device == "cpu":
        import torch

        torch.set_num_threads(args.threads)
    t_start = time.time()
    timings: dict[str, float] = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- records ---
    t0 = time.time()
    records = build_records(args.suffix, args.limit)
    timings["records"] = time.time() - t0

    # --- features ---
    if args.stage in ("all", "features"):
        t0 = time.time()
        frame = extract_features(
            records,
            suffix=args.suffix,
            device=args.device,
            base_batch_size=args.base_batch_size,
            claim_batch_size=args.claim_batch_size,
        )
        timings["features"] = time.time() - t0
    else:
        base = pd.read_parquet(DATA_PROCESSED / f"medhallu_base_features{args.suffix}.parquet")
        claims = pd.read_parquet(DATA_PROCESSED / f"medhallu_claim_features{args.suffix}.parquet")
        frame = base.merge(claims, on="sample_id", validate="one_to_one")
        frame = frame.merge(
            records[["sample_id", "pair_id", "role", "difficulty", "category", "n_knowledge_passages"]],
            on="sample_id",
            validate="one_to_one",
        )
        frame[SOURCE_COLUMN] = 1.0
        frame["item_idx"] = frame["item_idx"].astype(int)
        frame["label"] = frame["label"].astype(int)
        frame = frame.sort_values(["item_idx", "label"], ascending=[True, False]).reset_index(drop=True)

    if args.stage == "features":
        logger.info("Feature stage complete in %.1fs", time.time() - t_start)
        print(json.dumps({"rows": len(frame), "timings": timings}, indent=2))
        return

    # --- model scores ---
    t0 = time.time()
    scores_by_model: dict[str, dict] = {}
    for variant in ("m3", "m0"):
        scores_by_model[variant] = predict_variant(frame, variant)
        logger.info(
            "%s scores: %s",
            variant,
            {seed: round(float(np.mean(p)), 4) for seed, p in scores_by_model[variant].items()},
        )
    length_scores, length_info = length_only_baseline(frame)
    scores_by_model["length_only"] = {int(SEEDS[0]): length_scores}
    timings["predictions"] = time.time() - t0

    # --- metrics ---
    t0 = time.time()
    results: dict[str, dict] = {}
    csv_rows: list[dict] = []
    prediction_rows: list[dict] = []
    meta_cols = ["sample_id", "pair_id", "role", "difficulty", "category"]
    for model_name, by_seed in scores_by_model.items():
        per_seed = []
        for seed, scores in by_seed.items():
            bundle = score_frame(frame, scores)
            ci = pair_bootstrap(frame, scores, n_boot=args.n_boot)
            row = metric_row(bundle, ci)
            per_seed.append({"seed": int(seed), **row})
            for i, sample_id in enumerate(frame["sample_id"].to_numpy()):
                prediction_rows.append(
                    {
                        "dataset": "medhallu",
                        "model": model_name,
                        "seed": int(seed),
                        "sample_id": sample_id,
                        **frame.iloc[i][meta_cols[1:]].to_dict(),
                        "label": int(frame["label"].iloc[i]),
                        "score": float(scores[i]),
                    }
                )
            csv_rows.append(
                {
                    "dataset": "medhallu",
                    "model": model_name,
                    "seed": int(seed),
                    "n_features": N_FEATURES.get(model_name, 1),
                    **row,
                }
            )
        mean = {}
        for key in ("f1", "auroc", "pr_auc", "ece", "brier", "flagged_rate", "precision", "recall", "pair_accuracy"):
            values = [r[key] for r in per_seed]
            mean[key] = float(np.mean(values))
            mean[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        mean["n"] = per_seed[0]["n"]
        mean["n_pairs"] = per_seed[0]["n_pairs"]
        mean["threshold"] = THRESHOLD
        results[model_name] = {"per_seed": per_seed, "mean": mean}

        if len(by_seed) > 1:
            stacked = np.vstack([by_seed[s] for s in by_seed])
            avg = stacked.mean(axis=0)
            bundle = score_frame(frame, avg)
            ci = pair_bootstrap(frame, avg, n_boot=args.n_boot)
            row = metric_row(bundle, ci)
            results[model_name]["seed_average"] = row
            csv_rows.append(
                {
                    "dataset": "medhallu",
                    "model": f"{model_name}_seedavg",
                    "seed": -1,
                    "n_features": N_FEATURES[model_name],
                    **row,
                }
            )

    # Subgroups: difficulty and hallucination category (m3 seed-average scores).
    m3_avg = np.vstack([scores_by_model["m3"][s] for s in SEEDS]).mean(axis=0)
    m0_avg = np.vstack([scores_by_model["m0"][s] for s in SEEDS]).mean(axis=0)
    subgroups: dict[str, dict] = {}
    for model_name, avg in (("m3", m3_avg), ("m0", m0_avg)):
        model_sub = {}
        for group_col in ("difficulty", "category"):
            group_sub = {}
            for value in sorted(frame[group_col].unique()):
                mask = (frame[group_col] == value).to_numpy()
                sub = frame[mask].reset_index(drop=True)
                bundle = metric_bundle(sub["label"].to_numpy(), avg[mask], THRESHOLD)
                unique_pairs = pd.unique(sub["pair_id"])
                lookup = sub[["pair_id", "role"]].copy()
                lookup["row"] = np.arange(len(sub))
                gt = lookup[lookup.role == "grounded"].set_index("pair_id")["row"].reindex(unique_pairs).to_numpy()
                hal = lookup[lookup.role == "hallucinated"].set_index("pair_id")["row"].reindex(unique_pairs).to_numpy()
                group_sub[value] = {
                    "n_rows": int(len(sub)),
                    "n_pairs": int(len(unique_pairs)),
                    "f1": bundle["f1"],
                    "auroc": bundle["auroc"],
                    "pr_auc": bundle["pr_auc"],
                    "ece": bundle["ece"],
                    "brier": bundle["brier"],
                    "flagged_rate": bundle["predicted_positive_rate"],
                    "pair_accuracy": pair_accuracy(avg[mask], gt, hal),
                }
            model_sub[group_col] = group_sub
        subgroups[model_name] = model_sub
    timings["metrics"] = time.time() - t0

    # Sensitivity: m3 with the natural-response indicator forced to 0.
    m3_source0 = {}
    X0 = feature_matrix(frame.assign(**{SOURCE_COLUMN: 0.0}), m3_feature_names(), log_lengths=True)
    for seed in SEEDS:
        model = load_fixed_hp_model("m3", seed)
        scores = model.predict_proba(X0)[:, 1]
        bundle = score_frame(frame, scores)
        m3_source0[str(seed)] = metric_row(bundle)

    # HaluEval in-domain reference.
    reference = {} if args.skip_halueval_reference else halueval_reference()

    # --- payload ---
    elapsed = time.time() - t_start
    import sklearn
    import spacy
    import torch
    import xgboost
    import sentence_transformers

    payload = {
        "schema": "rebuild-medhallu-v1",
        "dataset": {
            "name": DATASET_LABEL,
            "source": "UTAustin-AIHealth/MedHallu",
            "config": "pqa_labeled",
            "license": "MIT",
            "raw_file": str(RAW.relative_to(ROOT)).replace("\\", "/"),
            "n_pairs": int(frame["pair_id"].nunique()),
            "n_rows": int(len(frame)),
            "rows_per_pair": 2,
            "label_balance": {str(int(k)): int(v) for k, v in frame["label"].value_counts().sort_index().items()},
            "domain": "biomedical question answering (PubMedQA-style evidence passages)",
            "annotation": "human-annotated ground-truth answers with LLM-written hallucinated answers",
            "used_for_training_or_tuning": False,
            "role": "second corpus, independent of HaluEval and RAGTruth",
            "construction": {
                "grounded_row": "answer = Ground Truth, label = 0",
                "hallucinated_row": "answer = Hallucinated Answer, label = 1",
                "question": "Question",
                "context": "'\\n\\n'.join(Knowledge)",
            },
            "difficulty_counts": {
                str(k): int(v) for k, v in frame.drop_duplicates("pair_id")["difficulty"].value_counts().items()
            },
            "category_counts": {
                str(k): int(v) for k, v in frame.drop_duplicates("pair_id")["category"].value_counts().items()
            },
            "answer_length_words": {
                "grounded": float(
                    records.loc[records["label"] == 0, "answer"].str.split().str.len().mean()
                ),
                "hallucinated": float(
                    records.loc[records["label"] == 1, "answer"].str.split().str.len().mean()
                ),
            },
        },
        "protocol": {
            "features": {
                "base": BASE_COLUMNS,
                "claim": CLAIM_COLUMNS,
                "n_base": len(BASE_COLUMNS),
                "n_claim": len(CLAIM_COLUMNS),
                "extractors": "src/features/extract_features.py + src/features/claim_features.py",
                "device": args.device,
                "precision": "fp16" if args.device == "cuda" else "fp32",
                "verdict_thresholds_file_present": (DATA_PROCESSED / "verdict_thresholds.json").exists(),
            },
            "models": {
                "m3": {
                    "description": "EC-XGB, 26 base + 8 claim + source indicator",
                    "n_features": N_FEATURES["m3"],
                    "log_lengths": True,
                    "source_indicator_value": 1.0,
                    "seeds": list(SEEDS),
                },
                "m0": {
                    "description": "standard XGBoost reference, 26 base features",
                    "n_features": N_FEATURES["m0"],
                    "log_lengths": False,
                    "seeds": list(SEEDS),
                },
                "length_only": {
                    "description": "logistic regression on log1p(n_words), fit on HaluEval train",
                    "train_rows": length_info["train_rows"],
                    "train_prevalence": length_info["train_prevalence"],
                    "coef": length_info["coef"],
                    "intercept": length_info["intercept"],
                    "halueval_train_mean_words": length_info["train_mean_words"],
                },
            },
            "decision_threshold": THRESHOLD,
            "bootstrap": {
                "n_boot": args.n_boot,
                "seed": PAIR_SEED,
                "unit": "pair (both rows drawn together)",
                "metrics": ["f1", "auroc", "pair_accuracy"],
            },
            "fixed_hp": {
                "max_depth": 5,
                "learning_rate": 0.05,
                "n_estimators": 300,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
            },
            "source_indicator_note": (
                "MedHallu is an external natural-response corpus; the deployed EC-XGB "
                "sets source_ragtruth=1.0 (src/api/main.py EC_SOURCE_VALUE), matching the "
                "B6 external runs. A sensitivity check with 0.0 is reported."
            ),
        },
        "results": results,
        "subgroups": subgroups,
        "sensitivity_m3_source_0": m3_source0,
        "halueval_reference": reference,
        "runtime": {
            "total_seconds": elapsed,
            "stage_seconds": timings,
            "python": sys.version.split()[0],
            "packages": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                "torch": torch.__version__,
                "spacy": spacy.__version__,
                "sentence-transformers": sentence_transformers.__version__,
            },
        },
        "limitations": [
            "Single second corpus; no cross-corpus pooling.",
            "Grounded answers are longer than hallucinated answers in MedHallu "
            f"(mean {records.loc[records['label'] == 0, 'answer'].str.split().str.len().mean():.1f} vs "
            f"{records.loc[records['label'] == 1, 'answer'].str.split().str.len().mean():.1f} words), "
            "the opposite direction of the HaluEval length artifact.",
            "One hallucinated answer per question, generated under the MedHallu protocol.",
            "Feature extractors and models were built for HaluEval; this is a zero-shot transfer test.",
            "The LLM-judge comparison on a 200-row subset was not run within the time budget.",
        ],
    }

    csv = pd.DataFrame(csv_rows)
    if not reference:
        pass
    else:
        ref_rows = []
        for variant, entry in reference.items():
            for row in entry["per_seed"]:
                ref_rows.append(
                    {
                        "dataset": "halueval_test",
                        "model": variant,
                        "seed": row["seed"],
                        "n_features": N_FEATURES[variant],
                        **{k: row[k] for k in (
                            "n", "prevalence", "threshold", "f1", "auroc", "pr_auc", "ece",
                            "brier", "flagged_rate", "precision", "recall",
                        )},
                    }
                )
            mean = entry["mean"]
            ref_rows.append(
                {
                    "dataset": "halueval_test",
                    "model": f"{variant}_mean",
                    "seed": -1,
                    "n_features": N_FEATURES[variant],
                    "n": entry["n"],
                    "prevalence": entry["per_seed"][0]["prevalence"],
                    "threshold": THRESHOLD,
                    **{k: mean[k] for k in (
                        "f1", "auroc", "pr_auc", "ece", "brier", "flagged_rate", "precision", "recall",
                    )},
                }
            )
        csv = pd.concat([csv, pd.DataFrame(ref_rows)], ignore_index=True)
    csv = csv.sort_values(["dataset", "model", "seed"]).reset_index(drop=True)
    out_csv = OUT_DIR / f"medhallu{args.out_suffix}.csv"
    csv.to_csv(out_csv, index=False)

    predictions = pd.DataFrame(prediction_rows)
    out_predictions = OUT_DIR / f"medhallu{args.out_suffix}_predictions.parquet"
    predictions.to_parquet(out_predictions, index=False)
    payload["predictions_file"] = str(out_predictions.relative_to(ROOT)).replace("\\", "/")

    out_json = OUT_DIR / f"medhallu{args.out_suffix}.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Console summary.
    cols = ["model", "seed", "f1", "auroc", "pr_auc", "ece", "brier", "flagged_rate", "pair_accuracy"]
    print("\nMedHallu metrics (threshold 0.5)")
    print(csv[csv["dataset"] == "medhallu"][cols].to_string(index=False))
    print(f"\nArtifacts: {out_csv} | {out_json}")
    print(f"Runtime: {elapsed:.1f}s total | stages: {timings}")


if __name__ == "__main__":
    main()
