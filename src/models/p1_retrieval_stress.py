"""P1: retrieval stress test for EC-XGB on RAGTruth QA.

The deployment claim is that the detector stays useful when the evidence comes
from retrieval instead of the gold passage. This script measures both halves of
that claim on the 900-row RAGTruth QA test split:

1. Retrieval quality. Build the existing hybrid index (BM25 + FAISS + RRF) over
   the 989 RAGTruth QA source documents, query it with each test question, and
   score recall@1/5/10 and MRR against the known source passage.
2. Downstream detection. Re-extract the 35 m3 features with the retrieved top-5
   context (gold features are read from ``external_features.parquet``), score
   both contexts with the three fixed-hp m3 models, and compare AUROC, F1, ECE,
   Brier, and the flag rate at threshold 0.5.

No model, threshold, or retrieval depth is tuned on the test split. The qrels
are the known RAGTruth source id, not human web-search relevance judgements.

Outputs (under ``artifacts/results/rebuild/p1``):
  retrieval_stress.csv
  retrieval_stress.json
  REPORT.md

Intermediate caches live under ``data/processed`` and are never treated as
results.

Run (repo root, .venv):
  python src/models/p1_retrieval_stress.py --stage all
  python src/models/p1_retrieval_stress.py --stage retrieval
  python src/models/p1_retrieval_stress.py --stage all --limit-rows 300
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.config import SEEDS  # noqa: E402
from src.models.rebuild_common import (  # noqa: E402
    ECE_BINS,
    EXTERNAL_FEATURES,
    FIXED_HP_PRED,
    draw_group_row_indices,
    f1_and_auroc,
    metric_bundle,
    m3_feature_names,
    summarize_bootstrap,
)
from src.models.run_b6_modified import SOURCE_COLUMN, feature_matrix  # noqa: E402
from src.models.train_pipeline import ece  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("p1_retrieval_stress")

# ---------------------------------------------------------------------------
# Paths and protocol constants
# ---------------------------------------------------------------------------
SOURCE_INFO = ROOT / "data" / "raw" / "ragtruth_official" / "source_info.jsonl"
UNIFIED = ROOT / "data" / "processed" / "unified_records.parquet"
MODEL_DIR = ROOT / "artifacts" / "models" / "rebuild" / "fixed_hp"
P1_DIR = ROOT / "artifacts" / "results" / "rebuild" / "p1"
INDEX_DIR = ROOT / "data" / "processed" / "retrieval_index_p1_qa"

RETRIEVAL_CACHE = ROOT / "data" / "processed" / "p1_retrieval_rows.parquet"
RETRIEVAL_CACHE_META = ROOT / "data" / "processed" / "p1_retrieval_rows.meta.json"
BASE_CACHE = ROOT / "data" / "processed" / "p1_retrieved_features.base.parquet"
CLAIMS_CACHE = ROOT / "data" / "processed" / "p1_retrieved_features.claims.parquet"
CLAIMS_PARTIAL = ROOT / "data" / "processed" / "p1_retrieved_features.claims.partial.parquet"
FEATURE_CACHE_META = ROOT / "data" / "processed" / "p1_retrieved_features.meta.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
TOP_K = 10          # retrieval depth scored by recall@10
CONTEXT_K = 5       # chunks passed to the detector
THRESHOLD = 0.5
N_BOOT = 1000
BOOT_SEED = 20261111
FEATURE_VERSION = "p1-features-v1"

METRIC_KEYS = (
    "auroc",
    "f1",
    "ece",
    "brier",
    "pr_auc",
    "precision",
    "recall",
    "predicted_positive_rate",
)
# PR-AUC is reported as a point estimate; the paired bootstrap tracks the
# headline discrimination/calibration terms and skips the extra ranking cost.
BOOT_METRIC_KEYS = tuple(key for key in METRIC_KEYS if key != "pr_auc")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _input_sha(question: str, context: str, answer: str) -> str:
    return _sha256(f"{question}\x00{context}\x00{answer}")[:20]


def _json_ready(value):
    """Recursively replace numpy scalars and non-finite floats for strict JSON."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _mean_std(values) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_qa_sources() -> pd.DataFrame:
    """The 989 RAGTruth QA source documents (id, question, passage text)."""
    records = []
    with SOURCE_INFO.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("task_type") != "QA":
                continue
            info = row.get("source_info", {})
            if isinstance(info, str):
                info = json.loads(info)
            records.append({
                "source_id": str(row["source_id"]),
                "source": str(row.get("source", "")),
                "source_question": str(info.get("question", "")),
                "passages": str(info.get("passages", "")),
            })
    frame = pd.DataFrame(records).sort_values("source_id").reset_index(drop=True)
    if len(frame) != 989:
        raise ValueError(f"expected 989 QA source documents, found {len(frame)}")
    if frame["passages"].duplicated().any():
        raise ValueError("duplicate QA passages: content-hash dedup would drop a gold source")
    return frame


def load_test_rows(limit_rows: int = 0, sample_mode: str = "random",
                   sample_seed: int = 42) -> pd.DataFrame:
    """RAGTruth QA test rows, optionally limited to complete source groups.

    Limited runs sample whole source groups (six answers each), not individual
    rows, so the six answers to one question stay together. The default is a
    deterministic random sample of groups rather than the first N ids, which
    would concentrate the subset in one block of the corpus.
    """
    frame = pd.read_parquet(UNIFIED)
    frame = frame[
        (frame["source_dataset"] == "ragtruth")
        & (frame["task"] == "qa")
        & (frame["official_split"] == "test")
    ].copy()
    if len(frame) != 900:
        raise ValueError(f"expected 900 RAGTruth QA test rows, found {len(frame)}")
    frame["gold_source_id"] = frame["source_group_id"].astype(str).str.split(":").str[-1]
    frame = frame.sort_values(["source_group_id", "sample_id"]).reset_index(drop=True)

    if limit_rows:
        n_groups = max(1, int(limit_rows) // 6)
        groups = np.array(sorted(frame["source_group_id"].unique()))
        if sample_mode == "random":
            rng = np.random.default_rng(sample_seed)
            chosen = rng.choice(groups, size=min(n_groups, len(groups)), replace=False)
        elif sample_mode == "head":
            chosen = groups[:n_groups]
        else:
            raise ValueError(f"unknown sample mode: {sample_mode}")
        frame = frame[frame["source_group_id"].isin(set(chosen))]
        frame = frame.sort_values(["source_group_id", "sample_id"]).reset_index(drop=True)
        logger.warning(
            "LIMITED RUN: %s rows from %s source groups (%s sample, seed %s; not the full "
            "900-row test split)", len(frame), frame["source_group_id"].nunique(),
            sample_mode, sample_seed,
        )
    return frame


# ---------------------------------------------------------------------------
# Retrieval index
# ---------------------------------------------------------------------------
def _make_embed_fn():
    """CPU-only MiniLM embedder with L2-normalised vectors (FAISS inner product)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    def embed(texts):
        vectors = model.encode(
            list(texts), batch_size=64, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    return embed


def _index_manifest(sources: pd.DataFrame) -> dict:
    from src.retrieval.chunk import CHUNK_SIZE, OVERLAP, chunk_text

    n_chunks = sum(len(chunk_text(text)) for text in sources["passages"])
    digest = _sha256("\x1f".join(sources["passages"].tolist()))
    return {
        "n_sources": int(len(sources)),
        "n_chunks": int(n_chunks),
        "passages_sha256": digest,
        "embedding_model": EMBEDDING_MODEL,
        "normalize_embeddings": True,
        "chunk_size": int(CHUNK_SIZE),
        "overlap": int(OVERLAP),
        "fusion": "RRF(k=60) over BM25 + FAISS inner product",
    }


def _attach_faiss(index) -> bool:
    """Attach the persisted FAISS vectors (``RetrievalIndex.load`` restores BM25 only).

    This keeps the run hybrid after a disk reload instead of silently falling
    back to BM25. Returns True when dense retrieval is available.
    """
    if index._faiss is not None:
        return True
    if not index.index_path.exists():
        return False
    import faiss

    reader = faiss.read_index(str(index.index_path))
    if reader.d != index._meta.get("dim") or reader.ntotal != len(index._passages):
        return False
    index._faiss = reader
    return True


def ensure_index(sources: pd.DataFrame, manifest: dict, fresh: bool = False):
    """Build or reload the disk index; returns (index, build_seconds, reused)."""
    from src.retrieval.index import RetrievalIndex

    marker_path = INDEX_DIR / "p1_manifest.json"
    embed = _make_embed_fn()

    if not fresh and marker_path.exists():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            marker = {}
        if marker.get("manifest") == manifest:
            index = RetrievalIndex(embed_fn=embed, index_dir=INDEX_DIR, top_k=CONTEXT_K)
            if index.status()["n_passages"] == manifest["n_chunks"] and _attach_faiss(index):
                logger.info("Reusing retrieval index (%s passages, dense=%s)",
                            index.status()["n_passages"], index._faiss is not None)
                return index, float(marker.get("build_seconds", 0.0)), True

    logger.info("Building retrieval index over %s QA sources...", len(sources))
    t0 = time.perf_counter()
    index = RetrievalIndex(embed_fn=embed, index_dir=INDEX_DIR, top_k=CONTEXT_K)
    index.clear()
    documents = [
        {"source": row.source_id, "url": f"ragtruth:{row.source_id}", "text": row.passages}
        for row in sources.itertuples(index=False)
    ]

    # ``add_documents`` embeds one document at a time (989 small encode calls).
    # Precompute all chunk vectors in one batched pass and pass a lookup function
    # so the build stays fast while still using the shared index code.
    from src.retrieval.chunk import chunk_text

    all_chunks = [chunk for doc in documents for chunk in chunk_text(doc["text"])]
    logger.info("Embedding %s chunks in one batch...", len(all_chunks))
    vectors = embed(all_chunks)
    lookup = {text: vec for text, vec in zip(all_chunks, vectors)}
    index.add_documents(documents, embed_fn=lambda texts: np.stack([lookup[t] for t in texts]))
    build_seconds = time.perf_counter() - t0

    n_passages = index.status()["n_passages"]
    covered = {p["source"] for p in index._passages}  # one-off audit of the private store
    if n_passages != manifest["n_chunks"]:
        raise ValueError(f"index has {n_passages} chunks, expected {manifest['n_chunks']}")
    if len(covered) != len(sources):
        missing = sorted(set(sources["source_id"]) - covered)
        raise ValueError(f"index does not cover {len(missing)} sources: {missing[:5]}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps({"manifest": manifest, "build_seconds": build_seconds,
                    "built_at_utc": pd.Timestamp.now("UTC").isoformat()}, indent=2),
        encoding="utf-8",
    )
    logger.info("Index built in %.1fs (%s chunks, %s sources)", build_seconds, n_passages, len(covered))
    return index, build_seconds, False


def _retrieval_cache_key(manifest: dict, frame: pd.DataFrame, limit_rows: int) -> dict:
    return {
        "schema": "p1-retrieval-v1",
        "index_manifest": manifest,
        "top_k": TOP_K,
        "context_k": CONTEXT_K,
        "limit_rows": int(limit_rows),
        "n_rows": int(len(frame)),
        "row_ids_sha256": _sha256("|".join(frame["sample_id"].astype(str).tolist())),
    }


def retrieve_rows(sources: pd.DataFrame, frame: pd.DataFrame, limit_rows: int,
                  fresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Query the hybrid index once per unique question and attach top-k context."""
    manifest = _index_manifest(sources)
    cache_key = _retrieval_cache_key(manifest, frame, limit_rows)

    if not fresh and RETRIEVAL_CACHE.exists() and RETRIEVAL_CACHE_META.exists():
        try:
            old_key = json.loads(RETRIEVAL_CACHE_META.read_text(encoding="utf-8"))
            cached = pd.read_parquet(RETRIEVAL_CACHE)
        except (OSError, ValueError):
            old_key, cached = {}, None
        if old_key == cache_key and cached is not None and len(cached) == len(frame):
            logger.info("Reusing retrieval cache: %s rows", len(cached))
            return cached, {"index_build_seconds": 0.0, "index_build_original_seconds": 0.0,
                            "search_seconds": 0.0, "index_reused": True,
                            "retrieval_reused": True, "manifest": manifest}

    index, build_seconds, index_reused = ensure_index(sources, manifest, fresh=fresh)

    unique_questions = frame.drop_duplicates("question")["question"].tolist()
    logger.info("Searching %s unique questions (top_k=%s)...", len(unique_questions), TOP_K)
    t0 = time.perf_counter()
    rankings: dict[str, list[dict]] = {}
    for question in unique_questions:
        rankings[question] = index.search(question, top_k=TOP_K)
    search_seconds = time.perf_counter() - t0

    rows = []
    for row in frame.itertuples(index=False):
        passages = rankings.get(row.question, [])
        sources_ranked = [str(p["source"]) for p in passages]
        gold = str(row.gold_source_id)
        first_rank = next((i + 1 for i, sid in enumerate(sources_ranked) if sid == gold), 0)
        context = "\n\n".join(str(p["text"]) for p in passages[:CONTEXT_K])
        rows.append({
            "sample_id": row.sample_id,
            "source_group_id": row.source_group_id,
            "gold_source_id": gold,
            "question": row.question,
            "answer": row.answer,
            "label": int(row.label),
            "retrieved_context": context,
            "top_sources": "|".join(sources_ranked),
            "top10_sources": "|".join(sources_ranked),
            "first_relevant_rank": int(first_rank),
            "hit@1": int(0 < first_rank <= 1),
            "hit@5": int(0 < first_rank <= 5),
            "hit@10": int(0 < first_rank <= 10),
            "n_relevant_top10": int(sum(sid == gold for sid in sources_ranked)),
            "top1_score": float(passages[0]["score"]) if passages else float("nan"),
            "context_chars": int(len(row.context)),
            "retrieved_chars": int(len(context)),
            "n_retrieved_chunks": int(min(CONTEXT_K, len(passages))),
        })
    out = pd.DataFrame(rows)
    out.to_parquet(RETRIEVAL_CACHE, index=False)
    RETRIEVAL_CACHE_META.write_text(json.dumps(cache_key, indent=2), encoding="utf-8")
    logger.info("Retrieval done in %.1fs; cache saved to %s", search_seconds, RETRIEVAL_CACHE)
    return out, {"index_build_seconds": 0.0 if index_reused else float(build_seconds),
                 "index_build_original_seconds": float(build_seconds),
                 "search_seconds": float(search_seconds), "index_reused": bool(index_reused),
                 "retrieval_reused": False, "manifest": manifest}


def retrieval_summary(frame: pd.DataFrame) -> dict:
    """Query-level recall@k and MRR; qrels are the known source document id."""
    queries = frame.drop_duplicates(["question", "gold_source_id"]).reset_index(drop=True)
    ranks = queries["first_relevant_rank"].to_numpy(dtype=int)
    mrr = float(np.mean(np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)))
    histogram = {}
    for rank in ranks:
        bucket = str(rank) if 0 < rank <= 10 else "miss"
        histogram[bucket] = histogram.get(bucket, 0) + 1
    ordered_histogram = {
        str(rank): histogram[str(rank)] for rank in range(1, 11) if str(rank) in histogram
    }
    if "miss" in histogram:
        ordered_histogram["miss"] = histogram["miss"]
    return {
        "n_queries": int(len(queries)),
        "n_rows": int(len(frame)),
        "n_gold_sources": int(queries["gold_source_id"].nunique()),
        "recall_at_1": float(queries["hit@1"].mean()),
        "recall_at_5": float(queries["hit@5"].mean()),
        "recall_at_10": float(queries["hit@10"].mean()),
        "mrr": mrr,
        "mean_relevant_chunks_top10": float(queries["n_relevant_top10"].mean()),
        "rank_histogram": ordered_histogram,
        "row_level_recall_at_5": float(frame["hit@5"].mean()),
        "row_level_mrr": float(np.mean(np.where(frame["first_relevant_rank"] > 0,
                                             1.0 / frame["first_relevant_rank"].clip(lower=1), 0.0))),
    }


# ---------------------------------------------------------------------------
# Feature extraction (retrieved context)
# ---------------------------------------------------------------------------
def _load_feature_cache(path: Path, inputs: pd.DataFrame, columns: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    cached = pd.read_parquet(path)
    if "input_sha" not in cached.columns:
        return None
    if any(column not in cached.columns for column in columns):
        return None
    expected = inputs.set_index("sample_id")["input_sha"]
    cached = cached[cached["sample_id"].isin(expected.index)].copy()
    cached = cached[cached["input_sha"] == cached["sample_id"].map(expected)]
    if cached.empty:
        return None
    return cached[["sample_id", "input_sha", *columns]]


def _feature_cache_meta(retrieval: pd.DataFrame, nli_name: str, args) -> dict:
    digest = _sha256("|".join(
        f"{r.sample_id}:{r.input_sha}" for r in retrieval.sort_values("sample_id").itertuples()
    ))
    return {
        "schema": FEATURE_VERSION,
        "retrieval_rows_sha256": digest,
        "n_rows": int(len(retrieval)),
        "nli_model": nli_name,
        "embedding_model": EMBEDDING_MODEL,
        "claim_params": {"max_context_sentences": 4, "max_claims": 6, "max_pair_chars": 400},
        "base_batch_size": int(args.base_batch_size),
        "claim_batch_size": int(args.claim_batch_size),
    }


def extract_retrieved_features(retrieval: pd.DataFrame, args) -> tuple[pd.DataFrame, dict]:
    """Extract 26 base + 8 claim features with the retrieved context (CPU)."""
    from src.features.claim_features import FEATURE_COLUMNS as CLAIM_COLUMNS
    from src.features.claim_features import extract_claim_features_df
    from src.features.extract_features import extract_full_feature_set, load_heavy_models

    base_columns = [c for c in m3_feature_names() if c != SOURCE_COLUMN and c not in CLAIM_COLUMNS]

    inputs = retrieval[["sample_id", "question", "retrieved_context", "answer", "label"]].copy()
    inputs["input_sha"] = [
        _input_sha(q, c, a) for q, c, a in zip(inputs["question"], inputs["retrieved_context"], inputs["answer"])
    ]
    inputs = inputs.rename(columns={"retrieved_context": "context"})

    t0 = time.perf_counter()
    models = load_heavy_models(device=args.device)
    load_seconds = time.perf_counter() - t0
    nli_name = models.get("nli_name", NLI_MODEL)
    if nli_name != NLI_MODEL:
        raise ValueError(f"unexpected NLI model: {nli_name} (frozen features use {NLI_MODEL})")

    meta = _feature_cache_meta(inputs, nli_name, args)
    old_meta = {}
    if FEATURE_CACHE_META.exists():
        try:
            old_meta = json.loads(FEATURE_CACHE_META.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old_meta = {}
    stable_keys = ("schema", "nli_model", "embedding_model", "claim_params",
                   "base_batch_size", "claim_batch_size")
    if args.fresh_features or (
        old_meta and any(old_meta.get(key) != meta[key] for key in stable_keys)
    ):
        for path in (BASE_CACHE, CLAIMS_CACHE, CLAIMS_PARTIAL):
            if path.exists():
                path.unlink()
                logger.info("Removed stale feature cache: %s", path)

    # --- chunked extraction: base + claims per 100 rows --------------------
    # Interleaving the two stages means an interrupted run leaves complete,
    # reusable rows instead of a base-only prefix that cannot be scored.
    cached_base = _load_feature_cache(BASE_CACHE, inputs, base_columns)
    cached_claims = _load_feature_cache(CLAIMS_CACHE, inputs, CLAIM_COLUMNS)
    base = cached_base.copy() if cached_base is not None else None
    claims = cached_claims.copy() if cached_claims is not None else None
    if cached_base is not None:
        logger.info("Reusing cached base features: %s rows", len(cached_base))
    if cached_claims is not None:
        logger.info("Reusing cached claim features: %s rows", len(cached_claims))

    sha_map = inputs.set_index("sample_id")["input_sha"]
    chunk_size = 100
    base_seconds = 0.0
    claim_seconds = 0.0
    computed_base = 0
    computed_claims = 0
    for start in range(0, len(inputs), chunk_size):
        chunk = inputs.iloc[start:start + chunk_size]
        base_done = set() if base is None else set(base["sample_id"])
        claims_done = set() if claims is None else set(claims["sample_id"])
        need_base = chunk[~chunk["sample_id"].isin(base_done)].copy()
        need_claims = chunk[~chunk["sample_id"].isin(claims_done)].copy()
        if need_base.empty and need_claims.empty:
            continue
        logger.info("Chunk %s-%s/%s: base %s rows, claims %s rows",
                    start + 1, start + len(chunk), len(inputs), len(need_base), len(need_claims))

        if not need_base.empty:
            t0 = time.perf_counter()
            extract_frame = need_base[["sample_id", "question", "context", "answer", "label"]].copy()
            extract_frame["item_idx"] = np.arange(len(extract_frame))
            extract_frame["split"] = "test"
            fresh_base = extract_full_feature_set(
                extract_frame, models, batch_size=args.base_batch_size
            )
            fresh_base["input_sha"] = fresh_base["sample_id"].map(sha_map)
            fresh_base = fresh_base[fresh_base["input_sha"].notna()]
            base = pd.concat(
                ([base] if base is not None else []) + [
                    fresh_base[["sample_id", "input_sha", *base_columns]]
                ],
                ignore_index=True,
            ).drop_duplicates("sample_id", keep="last")
            base.to_parquet(BASE_CACHE, index=False)
            base_seconds += time.perf_counter() - t0
            computed_base += len(need_base)
            logger.info("  base: %.1fs (%.2f rows/s, total %s rows)",
                        base_seconds, len(need_base) / max(time.perf_counter() - t0, 1e-9), len(base))

        if not need_claims.empty:
            t0 = time.perf_counter()
            claim_frame = need_claims[["sample_id", "question", "context", "answer"]].copy()
            fresh_claims = extract_claim_features_df(
                claim_frame, models["nli"], partial_path=CLAIMS_PARTIAL,
                checkpoint_every=50, batch_size=args.claim_batch_size, chunk_samples=24,
            )
            fresh_claims["input_sha"] = fresh_claims["sample_id"].map(sha_map)
            fresh_claims = fresh_claims[fresh_claims["input_sha"].notna()]
            claims = pd.concat(
                ([claims] if claims is not None else []) + [
                    fresh_claims[["sample_id", "input_sha", *CLAIM_COLUMNS]]
                ],
                ignore_index=True,
            ).drop_duplicates("sample_id", keep="last")
            claims.to_parquet(CLAIMS_CACHE, index=False)
            if CLAIMS_PARTIAL.exists():
                CLAIMS_PARTIAL.unlink()
            claim_seconds += time.perf_counter() - t0
            computed_claims += len(need_claims)
            logger.info("  claims: %.1fs (%.2f rows/s, total %s rows)",
                        claim_seconds, len(need_claims) / max(time.perf_counter() - t0, 1e-9), len(claims))

    if base is None or claims is None:
        raise RuntimeError("no feature rows were produced")

    FEATURE_CACHE_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    merged = retrieval.merge(base, on="sample_id", how="inner", validate="one_to_one")
    merged = merged.merge(claims, on="sample_id", how="inner", validate="one_to_one")
    missing = [c for c in m3_feature_names() if c != SOURCE_COLUMN and c not in merged.columns]
    if missing:
        raise ValueError(f"missing m3 features after extraction: {missing}")
    if len(merged) != len(inputs):
        raise ValueError(f"feature merge kept {len(merged)} of {len(inputs)} rows")
    if merged[[c for c in m3_feature_names() if c != SOURCE_COLUMN]].isna().any().any():
        raise ValueError("NaN in extracted m3 features")
    return merged, {
        "model_load_seconds": float(load_seconds),
        "base_seconds": float(base_seconds),
        "claim_seconds": float(claim_seconds),
        "base_reused": bool(cached_base is not None and computed_base == 0),
        "claims_reused": bool(cached_claims is not None and computed_claims == 0),
        "computed_base_rows": int(computed_base),
        "computed_claim_rows": int(computed_claims),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _align_gold(retrieval: pd.DataFrame) -> pd.DataFrame:
    gold = pd.read_parquet(EXTERNAL_FEATURES)
    gold = gold[
        (gold["source_dataset"] == "ragtruth")
        & (gold["task"] == "qa")
        & (gold["split"] == "test")
    ].copy()
    gold[SOURCE_COLUMN] = 1.0
    gold = gold[gold["sample_id"].isin(set(retrieval["sample_id"]))]
    if len(gold) != len(retrieval):
        raise ValueError(f"gold feature rows {len(gold)} != retrieval rows {len(retrieval)}")
    return gold.set_index("sample_id").loc[retrieval["sample_id"]].reset_index()


def score_models(retrieval: pd.DataFrame, gold: pd.DataFrame, retrieved: pd.DataFrame) -> dict:
    """Return raw probabilities for both contexts and every fixed-hp m3 seed."""
    import joblib

    names = m3_feature_names()
    frozen = pd.read_parquet(FIXED_HP_PRED)
    frozen = frozen[
        (frozen["variant"] == "m3") & (frozen["dataset"] == "ragtruth_qa_test")
    ]
    gold[SOURCE_COLUMN] = 1.0
    retrieved[SOURCE_COLUMN] = 1.0

    scores = {"gold": {}, "retrieved": {}}
    checks = {}
    for seed in SEEDS:
        model = joblib.load(MODEL_DIR / f"xgboost_m3_seed_{seed}.joblib")
        model.set_params(device="cpu")
        p_gold = model.predict_proba(feature_matrix(gold, names, log_lengths=True))[:, 1]
        p_ret = model.predict_proba(feature_matrix(retrieved, names, log_lengths=True))[:, 1]
        scores["gold"][seed] = p_gold
        scores["retrieved"][seed] = p_ret
        ref = frozen[frozen["seed"] == seed].set_index("sample_id").loc[gold["sample_id"], "score"].to_numpy()
        checks[f"seed_{seed}"] = {
            "frozen_gold_max_abs_diff": float(np.max(np.abs(p_gold - ref))),
            "n_frozen_rows": int(len(ref)),
        }
    return {"scores": scores, "checks": checks}


# ---------------------------------------------------------------------------
# Metrics and bootstrap
# ---------------------------------------------------------------------------
def _fast_bundle(y_true, y_score, threshold: float = THRESHOLD) -> dict:
    """Cheap metric set for the cluster bootstrap (same definitions as metric_bundle)."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.nan_to_num(np.asarray(y_score, dtype=float), nan=0.5), 1e-12, 1 - 1e-12)
    pred = p >= threshold
    tp = int(np.count_nonzero(pred & (y == 1)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero(~pred & (y == 1)))
    f1, auroc = f1_and_auroc(y, p, threshold)
    return {
        "auroc": auroc,
        "f1": f1,
        "ece": float(ece(y, p, ECE_BINS)),
        "brier": float(np.mean((p - y) ** 2)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "predicted_positive_rate": float(pred.mean()),
    }


def point_metrics(y_true, y_score) -> dict:
    bundle = metric_bundle(y_true, y_score, THRESHOLD)
    return {key: float(bundle[key]) for key in METRIC_KEYS}


def cluster_bootstrap(gold: pd.DataFrame, retrieved: pd.DataFrame, scores: dict,
                      n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Paired cluster bootstrap over the 150 source groups (six rows each)."""
    if len(gold) != len(retrieved):
        raise ValueError("gold/retrieved frames differ in length")
    g = gold["source_group_id"].astype(str).to_numpy()
    order = np.argsort(g, kind="stable")
    g_sorted = g[order]
    _, group_sizes = np.unique(g_sorted, return_counts=True)
    if len(group_sizes) < 5:
        return {"skipped": True, "reason": f"only {len(group_sizes)} groups"}

    y = gold["label"].to_numpy(dtype=int)[order]
    draws = {
        str(s): {
            context: {key: [] for key in BOOT_METRIC_KEYS}
            for context in ("gold", "retrieved", "delta")
        }
        for s in SEEDS
    }
    draws["mean"] = {context: {key: [] for key in BOOT_METRIC_KEYS}
                     for context in ("gold", "retrieved", "delta")}
    rng = np.random.default_rng(seed)

    for _ in range(n_boot):
        idx = draw_group_row_indices(group_sizes, rng)
        y_boot = y[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        per_seed = {"gold": {}, "retrieved": {}, "delta": {}}
        for s in SEEDS:
            b_gold = _fast_bundle(y_boot, scores["gold"][s][order][idx])
            b_ret = _fast_bundle(y_boot, scores["retrieved"][s][order][idx])
            per_seed["gold"][s] = b_gold
            per_seed["retrieved"][s] = b_ret
            per_seed["delta"][s] = {key: b_ret[key] - b_gold[key] for key in BOOT_METRIC_KEYS}
            for context in ("gold", "retrieved", "delta"):
                for key in BOOT_METRIC_KEYS:
                    draws[str(s)][context][key].append(per_seed[context][s][key])
        for context in ("gold", "retrieved", "delta"):
            for key in BOOT_METRIC_KEYS:
                draws["mean"][context][key].append(
                    float(np.nanmean([per_seed[context][s][key] for s in SEEDS]))
                )
    return {str(key): value for key, value in draws.items()}


def _summarize_draw_block(block: dict) -> dict:
    """Percentile summaries for one seed block (context -> metric -> draws)."""
    if block.get("skipped"):
        return {"skipped": True, "reason": block.get("reason")}
    return {
        context: {metric: summarize_bootstrap(vals) for metric, vals in values.items()}
        for context, values in block.items()
    }


def strata_metrics(gold: pd.DataFrame, retrieved: pd.DataFrame, scores: dict) -> dict:
    """Detection quality on rows whose gold source was / was not in the top-5."""
    out = {}
    for name, mask in (("hit5", retrieved["hit@5"].to_numpy(dtype=bool)),
                       ("miss5", ~retrieved["hit@5"].to_numpy(dtype=bool))):
        y = gold["label"].to_numpy(dtype=int)[mask]
        entry = {"n": int(mask.sum()), "n_groups": int(gold.loc[mask, "source_group_id"].nunique()),
                 "prevalence": float(y.mean()) if len(y) else float("nan")}
        if len(y) < 2 or len(np.unique(y)) < 2:
            entry["gold"] = {"skipped": True}
            entry["retrieved"] = {"skipped": True}
            out[name] = entry
            continue
        for context, source_scores in (("gold", scores["gold"]), ("retrieved", scores["retrieved"])):
            per_seed = [point_metrics(y, source_scores[s][mask]) for s in SEEDS]
            entry[context] = {
                metric: _mean_std([row[metric] for row in per_seed])[0] for metric in METRIC_KEYS
            }
        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "block", "context", "stratum", "seed", "n", "n_groups", "prevalence", "n_queries",
    "auroc", "auroc_std", "auroc_ci_low", "auroc_ci_high",
    "f1", "f1_std", "f1_ci_low", "f1_ci_high",
    "ece", "ece_std", "ece_ci_low", "ece_ci_high",
    "brier", "brier_std", "brier_ci_low", "brier_ci_high",
    "flagged_rate", "flagged_rate_std",
    "precision", "recall", "pr_auc",
    "recall_at_1", "recall_at_5", "recall_at_10", "mrr",
    "delta_auroc", "delta_auroc_ci_low", "delta_auroc_ci_high",
    "delta_f1", "delta_f1_ci_low", "delta_f1_ci_high",
    "delta_brier", "delta_brier_ci_low", "delta_brier_ci_high",
]


def _base_row(**overrides) -> dict:
    row = {column: np.nan for column in CSV_COLUMNS}
    row.update(overrides)
    return row


def build_csv(retrieval: dict, metrics: dict, boot: dict, strata: dict) -> pd.DataFrame:
    rows = []
    rows.append(_base_row(
        block="retrieval", context="hybrid_top10", seed="", n=retrieval["n_rows"],
        n_groups=retrieval["n_gold_sources"], n_queries=retrieval["n_queries"],
        recall_at_1=retrieval["recall_at_1"], recall_at_5=retrieval["recall_at_5"],
        recall_at_10=retrieval["recall_at_10"], mrr=retrieval["mrr"],
    ))

    for context in ("gold", "retrieved"):
        for seed in SEEDS:
            bundle = metrics["per_seed"][context][str(seed)]
            ci = boot.get(str(seed), {}).get(context, {})
            rows.append(_base_row(
                block="model", context=context, seed=seed,
                n=metrics["n"], n_groups=metrics["n_groups"],
                prevalence=metrics["prevalence"],
                auroc=bundle["auroc"], auroc_ci_low=ci.get("auroc", {}).get("q025", np.nan),
                auroc_ci_high=ci.get("auroc", {}).get("q975", np.nan),
                f1=bundle["f1"], f1_ci_low=ci.get("f1", {}).get("q025", np.nan),
                f1_ci_high=ci.get("f1", {}).get("q975", np.nan),
                ece=bundle["ece"], ece_ci_low=ci.get("ece", {}).get("q025", np.nan),
                ece_ci_high=ci.get("ece", {}).get("q975", np.nan),
                brier=bundle["brier"], brier_ci_low=ci.get("brier", {}).get("q025", np.nan),
                brier_ci_high=ci.get("brier", {}).get("q975", np.nan),
                flagged_rate=bundle["predicted_positive_rate"],
                precision=bundle["precision"], recall=bundle["recall"], pr_auc=bundle["pr_auc"],
            ))
        summary = metrics["summary"][context]
        ci = boot.get("mean", {}).get(context, {})
        rows.append(_base_row(
            block="model", context=context, seed="mean",
            n=metrics["n"], n_groups=metrics["n_groups"], prevalence=metrics["prevalence"],
            auroc=summary["auroc"][0], auroc_std=summary["auroc"][1],
            auroc_ci_low=ci.get("auroc", {}).get("q025", np.nan),
            auroc_ci_high=ci.get("auroc", {}).get("q975", np.nan),
            f1=summary["f1"][0], f1_std=summary["f1"][1],
            f1_ci_low=ci.get("f1", {}).get("q025", np.nan),
            f1_ci_high=ci.get("f1", {}).get("q975", np.nan),
            ece=summary["ece"][0], ece_std=summary["ece"][1],
            ece_ci_low=ci.get("ece", {}).get("q025", np.nan),
            ece_ci_high=ci.get("ece", {}).get("q975", np.nan),
            brier=summary["brier"][0], brier_std=summary["brier"][1],
            brier_ci_low=ci.get("brier", {}).get("q025", np.nan),
            brier_ci_high=ci.get("brier", {}).get("q975", np.nan),
            flagged_rate=summary["predicted_positive_rate"][0],
            flagged_rate_std=summary["predicted_positive_rate"][1],
            precision=summary["precision"][0], recall=summary["recall"][0],
            pr_auc=summary["pr_auc"][0],
        ))

    for seed in [*SEEDS, "mean"]:
        key = str(seed)
        values = {"seed": seed, "n": metrics["n"], "n_groups": metrics["n_groups"]}
        for metric in ("auroc", "f1", "brier"):
            ci = boot.get(key, {}).get("delta", {}).get(metric, {})
            values[f"delta_{metric}"] = metrics["delta"][key][metric]
            values[f"delta_{metric}_ci_low"] = ci.get("q025", np.nan)
            values[f"delta_{metric}_ci_high"] = ci.get("q975", np.nan)
        rows.append(_base_row(block="delta", context="retrieved_minus_gold", **values))

    for name, entry in strata.items():
        for context in ("gold", "retrieved"):
            values = entry.get(context, {})
            rows.append(_base_row(
                block="stratum", context=context, stratum=name, seed="mean",
                n=entry["n"], n_groups=entry["n_groups"], prevalence=entry["prevalence"],
                auroc=values.get("auroc", np.nan), f1=values.get("f1", np.nan),
                ece=values.get("ece", np.nan), brier=values.get("brier", np.nan),
                flagged_rate=values.get("predicted_positive_rate", np.nan),
                precision=values.get("precision", np.nan), recall=values.get("recall", np.nan),
                pr_auc=values.get("pr_auc", np.nan),
            ))
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(path: Path, data: dict) -> None:
    ret = data["retrieval"]
    met = data["metrics"]
    boot = data["bootstrap"]
    runtime = data["runtime_seconds"]
    checks = data["checks"]
    data_info = data["data"]

    def row(context: str, seed: str) -> str:
        if seed == "mean":
            m = {key: met["summary"][context][key][0] for key in METRIC_KEYS}
        else:
            m = met["per_seed"][context][str(int(seed))]
        return (
            f"| {context} | {seed} | {_fmt(m['auroc'])} | {_fmt(m['f1'])} | "
            f"{_fmt(m['ece'])} | {_fmt(m['brier'])} | {_fmt(m['predicted_positive_rate'])} | "
            f"{_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['pr_auc'])} |"
        )

    lines = []
    lines.append("# P1 - Retrieval Stress Test on RAGTruth QA")
    lines.append("")
    lines.append("Generated by `src/models/p1_retrieval_stress.py`. Every number below comes from")
    lines.append("`retrieval_stress.csv` / `retrieval_stress.json`; none are hand-entered.")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append("Does EC-XGB keep its detection quality when the context comes from the")
    lines.append("project's hybrid retriever instead of the gold RAGTruth passage? The test")
    lines.append("separates retrieval quality from detector degradation.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- Corpus: {data_info['n_sources']:,} RAGTruth QA source documents "
                 f"({data_info['n_chunks']:,} chunks after the shared 1,500/200 character chunker).")
    lines.append(f"- Index: BM25 + FAISS inner product over CPU MiniLM "
                 f"(`{EMBEDDING_MODEL}`), fused with RRF (k=60); the same `RetrievalIndex` "
                 "the API uses.")
    lines.append(f"- Queries: {data_info['n_test_questions']} unique test questions "
                 f"({data_info['n_test_rows']} rows, six answers per source).")
    lines.append(f"- Context condition: concatenated top-{CONTEXT_K} retrieved chunks.")
    lines.append(f"- Detector: m3, 35 features, fixed hyperparameters "
                 f"(max_depth=5, learning_rate=0.05, n_estimators=300, subsample=0.7, "
                 f"colsample_bytree=0.7), seeds 42/123/456, threshold {THRESHOLD}.")
    lines.append(f"- Gold features: `external_features.parquet`; retrieved features re-extracted "
                 f"with the same code on {data_info['device']}.")
    lines.append(f"- Intervals: {N_BOOT:,}-draw paired cluster bootstrap over source groups "
                 f"({data_info['n_test_groups']} groups of six rows).")
    lines.append("- No model, threshold, feature, or retrieval depth was tuned on the test split.")
    lines.append("")
    lines.append("## Retrieval results")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| Recall@1 | {_fmt(ret['recall_at_1'])} |")
    lines.append(f"| Recall@5 | {_fmt(ret['recall_at_5'])} |")
    lines.append(f"| Recall@10 | {_fmt(ret['recall_at_10'])} |")
    lines.append(f"| MRR | {_fmt(ret['mrr'])} |")
    lines.append(f"| Mean relevant chunks in top-10 | {_fmt(ret['mean_relevant_chunks_top10'])} |")
    lines.append(f"| Queries | {ret['n_queries']} |")
    lines.append("")
    hist = ret["rank_histogram"]
    lines.append("Rank of the first gold-source chunk: " + ", ".join(
        f"rank {rank}: {count}" for rank, count in hist.items()) + ".")
    lines.append("")
    lines.append("## Detection results")
    lines.append("")
    lines.append("| context | seed | AUROC | F1@0.5 | ECE | Brier | flagged | precision | recall | PR-AUC |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for seed in [*SEEDS, "mean"]:
        for context in ("gold", "retrieved"):
            lines.append(row(context, str(seed)))
    lines.append("")
    lines.append("Seed SD (AUROC / F1 / ECE / Brier): gold "
                 f"{_fmt(met['summary']['gold']['auroc'][1])} / "
                 f"{_fmt(met['summary']['gold']['f1'][1])} / "
                 f"{_fmt(met['summary']['gold']['ece'][1])} / "
                 f"{_fmt(met['summary']['gold']['brier'][1])}; retrieved "
                 f"{_fmt(met['summary']['retrieved']['auroc'][1])} / "
                 f"{_fmt(met['summary']['retrieved']['f1'][1])} / "
                 f"{_fmt(met['summary']['retrieved']['ece'][1])} / "
                 f"{_fmt(met['summary']['retrieved']['brier'][1])}.")
    lines.append("")
    lines.append("## Retrieved minus gold")
    lines.append("")
    lines.append("| seed | dAUROC | 95% CI | dF1 | 95% CI | dBrier | 95% CI |")
    lines.append("|---|---:|---|---:|---|---:|---|")
    for seed in [*SEEDS, "mean"]:
        delta = met["delta"][str(seed)]
        interval = boot.get(str(seed), {}).get("delta", {})
        cells = []
        for metric in ("auroc", "f1", "brier"):
            ci = interval.get(metric, {})
            cells.extend([
                _fmt(delta[metric]),
                f"[{_fmt(ci.get('q025'))}, {_fmt(ci.get('q975'))}]",
            ])
        lines.append(f"| {seed} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | "
                     f"{cells[4]} | {cells[5]} |")
    lines.append("")
    lines.append("## Reading the result")
    lines.append("")
    mean_ci = boot.get("mean", {}).get("delta", {})
    auroc_ci = mean_ci.get("auroc", {})
    f1_ci = mean_ci.get("f1", {})
    hits = sum(count for rank, count in ret["rank_histogram"].items()
               if rank != "miss" and int(rank) <= 5)
    lines.append(f"- The retriever puts the gold source in the top-5 for {hits} of "
                 f"{ret['n_queries']} questions (recall@5 {_fmt(ret['recall_at_5'])}); "
                 f"MRR is {_fmt(ret['mrr'])}.")
    lines.append(f"- Seed-averaged AUROC moves from {_fmt(met['summary']['gold']['auroc'][0])} "
                 f"with gold context to {_fmt(met['summary']['retrieved']['auroc'][0])} with "
                 f"retrieved context, a change of {_fmt(met['delta']['mean']['auroc'])} "
                 f"[{_fmt(auroc_ci.get('q025'))}, {_fmt(auroc_ci.get('q975'))}].")
    auroc_excludes_zero = (
        np.isfinite(auroc_ci.get("q025", np.nan)) and np.isfinite(auroc_ci.get("q975", np.nan))
        and (auroc_ci["q025"] > 0 or auroc_ci["q975"] < 0)
    )
    if auroc_excludes_zero:
        lines.append("  The paired cluster interval excludes zero, so the AUROC drop is larger "
                     "than the resampling noise in this test.")
    else:
        lines.append("  The paired cluster interval includes zero, so this test does not resolve "
                     "the AUROC change from sampling noise.")
    lines.append(f"- F1 at 0.5 changes by {_fmt(met['delta']['mean']['f1'])} "
                 f"[{_fmt(f1_ci.get('q025'))}, {_fmt(f1_ci.get('q975'))}] while the flag rate "
                 f"moves from {_fmt(met['summary']['gold']['predicted_positive_rate'][0])} to "
                 f"{_fmt(met['summary']['retrieved']['predicted_positive_rate'][0])}. At this "
                 "operating point F1 mostly tracks the flag rate, so AUROC is the cleaner "
                 "discrimination statement.")
    lines.append(f"- ECE moves from {_fmt(met['summary']['gold']['ece'][0])} to "
                 f"{_fmt(met['summary']['retrieved']['ece'][0])} and Brier from "
                 f"{_fmt(met['summary']['gold']['brier'][0])} to "
                 f"{_fmt(met['summary']['retrieved']['brier'][0])}; both contexts are far from "
                 "calibrated, so retrieval does not repair the score scale.")
    miss = data["strata"].get("miss5", {})
    if miss.get("n", 0):
        lines.append(f"- The miss5 stratum contains {miss['n']} rows from "
                     f"{miss['n_groups']} source groups. Its metrics are reported above; the "
                     "cell is small, so treat it as a case description rather than an estimate.")
    else:
        lines.append("- No question missed the top-5, so this run stresses distractor noise and "
                     "context length but not the complete-absence-of-evidence case.")
    gold_median = data["context_lengths"]["gold"]["median_chars"]
    ret_median = data["context_lengths"]["retrieved_top5"]["median_chars"]
    if gold_median:
        lines.append(f"- The retrieved context is {ret_median / gold_median:.1f}x longer at the "
                     f"median ({ret_median:.0f} vs {gold_median:.0f} characters). The NLI base "
                     "features truncate at 512 tokens, so they read only the opening of the "
                     "retrieved context; the claim features search the full context.")
    lines.append("")
    lines.append("## Where retrieval failed")
    lines.append("")
    lines.append("Rows are split by whether the gold source appears in the top-5 retrieved")
    lines.append("chunks. Metrics are seed-averaged; counts are row counts.")
    lines.append("")
    lines.append("| stratum | context | n | groups | prevalence | AUROC | F1@0.5 | ECE | Brier | flagged |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, entry in data["strata"].items():
        for context in ("gold", "retrieved"):
            values = entry.get(context, {})
            lines.append(
                f"| {name} | {context} | {entry['n']} | {entry['n_groups']} | "
                f"{_fmt(entry['prevalence'])} | {_fmt(values.get('auroc'))} | "
                f"{_fmt(values.get('f1'))} | {_fmt(values.get('ece'))} | "
                f"{_fmt(values.get('brier'))} | {_fmt(values.get('predicted_positive_rate'))} |"
            )
    lines.append("")
    lines.append("## Context length")
    lines.append("")
    lines.append("| context | mean chars | median chars | min | max |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, values in data["context_lengths"].items():
        lines.append(f"| {label} | {_fmt(values['mean_chars'], 1)} | "
                     f"{_fmt(values['median_chars'], 1)} | {values['min_chars']} | {values['max_chars']} |")
    lines.append("")
    lines.append("## Checks and provenance")
    lines.append("")
    lines.append(f"- Gold probabilities reproduce the frozen fixed-hp table: max absolute "
                 f"difference {checks['frozen_gold_max_abs_diff']:.2e} across seeds.")
    lines.append(f"- Gold features: `data/processed/external_features.parquet`; retrieved "
                 f"features: `data/processed/p1_retrieved_features.*` (cache).")
    lines.append(f"- Fast bootstrap metric vs `metric_bundle` max difference: "
                 f"{checks['fast_metric_max_abs_diff']:.2e}.")
    lines.append(f"- Retrieval index reused: {data_info['index_reused']}; retrieval cache reused: "
                 f"{data_info['retrieval_reused']}.")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append("| stage | seconds |")
    lines.append("|---|---:|")
    for stage, seconds in runtime.items():
        lines.append(f"| {stage} | {seconds:.1f} |")
    lines.append(f"| total | {data['runtime_total_seconds']:.1f} |")
    lines.append("")
    if "index_build_original" in runtime:
        lines.append("`index_build` is 0 because the index was reused; `index_build_original` is "
                     "the one-off build time recorded in the index marker.")
        lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- The qrels are the known RAGTruth source passage (`source_id`), not human")
    lines.append("  web-search relevance judgements. This is a controlled corpus stress test,")
    lines.append("  not a measure of open-web retrieval quality.")
    lines.append("- The index contains all 989 QA sources, including the sources behind the")
    lines.append("  test rows. Retrieval is answer-independent: all six answers to one question")
    lines.append("  share the same retrieved context and the same qrels.")
    lines.append("- The NLI base features see only the first 512 tokens of the context because of")
    lines.append("  the cross-encoder limit, while the claim features search sentences across the")
    lines.append("  full retrieved context. Retrieved contexts are longer than gold contexts, so")
    lines.append("  part of the degradation is a truncation/noise effect, not only missing evidence.")
    lines.append("- RAGTruth QA is hard for this model even with gold context (AUROC near chance);")
    lines.append("  the comparison is about the direction and size of the retrieval shift.")
    lines.append("- Flag rates are high at threshold 0.5, so F1 at that operating point mostly")
    lines.append("  tracks the flag rate. AUROC is the threshold-free headline.")
    lines.append("- CIs cover sampling and seed variability only; they do not cover retriever or")
    lines.append("  chunker selection uncertainty.")
    if data_info.get("limited"):
        lines.append(f"- LIMITED RUN: {data_info['n_test_rows']} of 900 test rows used "
                     f"({data_info['n_test_groups']} source groups, "
                     f"{data_info.get('sample_mode')} sample, seed "
                     f"{data_info.get('sample_seed')}).")
    lines.append("")
    lines.append("## Blockers")
    lines.append("")
    if data["blockers"]:
        for blocker in data["blockers"]:
            lines.append(f"- {blocker}")
    elif data_info.get("limited"):
        lines.append(f"- None for this {data_info['n_test_rows']}-row subset. The full 900-row "
                     "run was not completed; the subset is the reported evidence.")
    else:
        lines.append("- None. The full 900-row test split completed.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="P1 retrieval stress test (RAGTruth QA)")
    parser.add_argument("--stage", choices=("retrieval", "features", "report", "all"), default="all")
    parser.add_argument("--limit-rows", type=int, default=0,
                        help="Use only the first N rows (complete source groups). 0 = all 900.")
    parser.add_argument("--sample-mode", choices=("random", "head"), default="random",
                        help="How to pick source groups for a limited run.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--base-batch-size", type=int, default=32)
    parser.add_argument("--claim-batch-size", type=int, default=32)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--fresh-index", action="store_true")
    parser.add_argument("--fresh-retrieval", action="store_true")
    parser.add_argument("--fresh-features", action="store_true")
    args = parser.parse_args()

    t_start = time.perf_counter()
    P1_DIR.mkdir(parents=True, exist_ok=True)
    runtime = {}

    if args.stage == "report":
        bundle_path = P1_DIR / "retrieval_stress.json"
        if not bundle_path.exists():
            raise FileNotFoundError(f"{bundle_path} not found; run --stage all first")
        data = json.loads(bundle_path.read_text(encoding="utf-8"))
        write_report(P1_DIR / "REPORT.md", data)
        print(f"Rewrote {P1_DIR / 'REPORT.md'} from {bundle_path.name}")
        return

    sources = load_qa_sources()
    frame = load_test_rows(args.limit_rows, args.sample_mode, args.sample_seed)
    retrieval, retrieval_runtime = retrieve_rows(
        sources, frame, args.limit_rows, fresh=args.fresh_retrieval or args.fresh_index
    )
    ret_summary = retrieval_summary(retrieval)
    runtime["index_build"] = retrieval_runtime["index_build_seconds"]
    if retrieval_runtime["index_build_original_seconds"] > 0:
        runtime["index_build_original"] = retrieval_runtime["index_build_original_seconds"]
    runtime["retrieval_search"] = retrieval_runtime["search_seconds"]
    logger.info("Retrieval: R@1=%.3f R@5=%.3f R@10=%.3f MRR=%.3f",
                ret_summary["recall_at_1"], ret_summary["recall_at_5"],
                ret_summary["recall_at_10"], ret_summary["mrr"])

    if args.stage == "retrieval":
        print(json.dumps(_json_ready(ret_summary), indent=2))
        return

    gold = _align_gold(retrieval)
    retrieved, feature_runtime = extract_retrieved_features(retrieval, args)
    retrieved = retrieved.set_index("sample_id").loc[retrieval["sample_id"]].reset_index()
    runtime.update({
        "feature_model_load": feature_runtime["model_load_seconds"],
        "feature_base": feature_runtime["base_seconds"],
        "feature_claims": feature_runtime["claim_seconds"],
    })

    t0 = time.perf_counter()
    scored = score_models(retrieval, gold, retrieved)
    scores = scored["scores"]
    runtime["prediction"] = time.perf_counter() - t0

    # Point metrics
    y = gold["label"].to_numpy(dtype=int)
    per_seed = {"gold": {}, "retrieved": {}}
    for context in ("gold", "retrieved"):
        for seed in SEEDS:
            per_seed[context][seed] = point_metrics(y, scores[context][seed])
    summary = {}
    for context in ("gold", "retrieved"):
        summary[context] = {key: _mean_std([per_seed[context][s][key] for s in SEEDS])
                            for key in METRIC_KEYS}
    delta = {}
    for seed in [*SEEDS, "mean"]:
        if seed == "mean":
            delta[str(seed)] = {
                key: float(np.mean([per_seed["retrieved"][s][key] - per_seed["gold"][s][key]
                                    for s in SEEDS]))
                for key in METRIC_KEYS
            }
        else:
            delta[str(seed)] = {key: per_seed["retrieved"][seed][key] - per_seed["gold"][seed][key]
                                for key in METRIC_KEYS}

    t0 = time.perf_counter()
    boot = cluster_bootstrap(gold, retrieved, scores, n_boot=args.n_boot)
    runtime["bootstrap"] = time.perf_counter() - t0
    boot_summary = {str(key): _summarize_draw_block(val) for key, val in boot.items()} if boot else {}

    strata = strata_metrics(gold, retrieved, scores)

    # Fast metric check: bootstrap helper vs official bundle on the first seed.
    fast = _fast_bundle(y, scores["gold"][SEEDS[0]])
    official = point_metrics(y, scores["gold"][SEEDS[0]])
    fast_diff = max(abs(fast[k] - official[k]) for k in fast)

    metrics = {
        "n": int(len(y)),
        "n_groups": int(gold["source_group_id"].nunique()),
        "prevalence": float(y.mean()),
        "per_seed": {context: {str(s): per_seed[context][s] for s in SEEDS}
                     for context in ("gold", "retrieved")},
        "summary": {context: {key: list(summary[context][key]) for key in METRIC_KEYS}
                    for context in ("gold", "retrieved")},
        "delta": delta,
    }

    context_lengths = {
        "gold": {
            "mean_chars": float(retrieval["context_chars"].mean()),
            "median_chars": float(retrieval["context_chars"].median()),
            "min_chars": int(retrieval["context_chars"].min()),
            "max_chars": int(retrieval["context_chars"].max()),
        },
        "retrieved_top5": {
            "mean_chars": float(retrieval["retrieved_chars"].mean()),
            "median_chars": float(retrieval["retrieved_chars"].median()),
            "min_chars": int(retrieval["retrieved_chars"].min()),
            "max_chars": int(retrieval["retrieved_chars"].max()),
        },
    }

    frozen_diff = max(v["frozen_gold_max_abs_diff"] for v in scored["checks"].values())
    blockers = []
    if args.limit_rows:
        blockers.append(
            f"Full 900-row extraction was not completed. Reported evidence is a "
            f"{len(retrieval)}-row {args.sample_mode} sample of "
            f"{retrieval['source_group_id'].nunique()} source groups "
            f"(sample seed {args.sample_seed})."
        )
    data = {
        "schema": "p1-retrieval-stress-v1",
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "script": "src/models/p1_retrieval_stress.py",
        "protocol": {
            "embedding_model": EMBEDDING_MODEL,
            "retrieval": "BM25 + FAISS inner product, RRF k=60",
            "top_k_scored": TOP_K,
            "context_k": CONTEXT_K,
            "model": "m3 (EC-XGB), fixed hyperparameters",
            "seeds": SEEDS,
            "threshold": THRESHOLD,
            "n_bootstrap": int(args.n_boot),
            "bootstrap": "paired cluster bootstrap over source_group_id",
            "qrels": "known RAGTruth source passage (source_id), not human web-search qrels",
            "test_tuning": "none",
        },
        "data": {
            "n_sources": int(len(sources)),
            "n_chunks": int(retrieval_runtime["manifest"]["n_chunks"]),
            "n_test_rows": int(len(retrieval)),
            "n_test_questions": int(retrieval["question"].nunique()),
            "n_test_groups": int(retrieval["source_group_id"].nunique()),
            "n_features": int(len(m3_feature_names())),
            "device": args.device,
            "limited": bool(args.limit_rows),
            "sample_mode": args.sample_mode if args.limit_rows else "all",
            "sample_seed": int(args.sample_seed) if args.limit_rows else None,
            "index_reused": bool(retrieval_runtime["index_reused"]),
            "retrieval_reused": bool(retrieval_runtime["retrieval_reused"]),
        },
        "retrieval": ret_summary,
        "metrics": metrics,
        "bootstrap": boot_summary,
        "strata": strata,
        "context_lengths": context_lengths,
        "checks": {
            "frozen_gold_max_abs_diff": float(frozen_diff),
            "frozen_per_seed": scored["checks"],
            "fast_metric_max_abs_diff": float(fast_diff),
            "feature_cache": {
                "base_reused": feature_runtime["base_reused"],
                "claims_reused": feature_runtime["claims_reused"],
                "computed_base_rows": feature_runtime["computed_base_rows"],
                "computed_claim_rows": feature_runtime["computed_claim_rows"],
            },
        },
        "runtime_seconds": runtime,
        "blockers": blockers,
    }
    data["runtime_total_seconds"] = time.perf_counter() - t_start
    data = _json_ready(data)

    csv = build_csv(ret_summary, metrics, boot_summary, strata)
    csv.to_csv(P1_DIR / "retrieval_stress.csv", index=False)
    (P1_DIR / "retrieval_stress.json").write_text(json.dumps(data, indent=2, allow_nan=False),
                                                 encoding="utf-8")
    write_report(P1_DIR / "REPORT.md", data)

    print("\nRetrieval:", {k: round(v, 4) for k, v in ret_summary.items()
                           if isinstance(v, float)})
    print("\nDetection (mean over seeds):")
    for context in ("gold", "retrieved"):
        print(f"  {context:9s}", {k: round(summary[context][k][0], 4) for k in METRIC_KEYS})
    print("  delta    ", {k: round(delta['mean'][k], 4) for k in METRIC_KEYS})
    print(f"\nWrote {P1_DIR}")


if __name__ == "__main__":
    main()
