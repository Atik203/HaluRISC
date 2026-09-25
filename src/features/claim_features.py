"""
M2: claim-level NLI aggregation features for the modified XGBoost variant.

For every sample the answer is split into atomic claims (src/claims/decompose)
and each claim is scored with the NLI cross-encoder against the context
sentences. The per-(claim, sentence) probabilities are aggregated into eight
features that summarize how well the evidence supports the answer claim by
claim. The deployed system computes these signals for display; here they
become model features for the b6 variant.

Features (8):
  claim_max_contradicts     max P(contradiction) over all (claim, sentence) pairs
  claim_mean_contradicts    mean of the per-claim max contradiction
  claim_contradicted_ratio  fraction of claims whose best verdict is contradicted
  claim_supported_ratio     fraction of claims whose best verdict is supported
  claim_unsupported_ratio   fraction of claims with no verdict at the 0.5 cutoff
  claim_min_entails         min over claims of the per-claim max entailment
  claim_mean_entails        mean of the per-claim max entailment
  n_claims                  number of atomic claims in the answer

The verdict cutoffs match src/claims/verify.py (entail >= 0.5, contra >= 0.5).
Empty context uses the neutral 1/3 row, exactly like the base NLI features.

Run (repo root, .venv):
  python src/features/claim_features.py                    # full qa_clean (20k)
  python src/features/claim_features.py --limit 20         # smoke test
  python src/features/claim_features.py --input data/processed/qa_clean.parquet

The run is resumable: progress is checkpointed to <output>.partial.parquet
every --checkpoint-every samples and already-computed sample_ids are skipped.
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.claims.decompose import split_claims, split_sentences
from src.features.nli_features import LABELS, NEUTRAL, load_nli_model
from src.models.config import DATA_PROCESSED

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("claim_features")

FEATURE_COLUMNS = [
    "claim_max_contradicts",
    "claim_mean_contradicts",
    "claim_contradicted_ratio",
    "claim_supported_ratio",
    "claim_unsupported_ratio",
    "claim_min_entails",
    "claim_mean_entails",
    "n_claims",
]

ENTAIL_CUTOFF = 0.5
CONTRA_CUTOFF = 0.5
MAX_CONTEXT_SENTENCES = 8
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))

DEFAULT_INPUT = DATA_PROCESSED / "qa_clean.parquet"
DEFAULT_OUTPUT = DATA_PROCESSED / "claim_features.parquet"


def _empty_row(n_claims: int = 0) -> dict:
    """No context: mirror the neutral NLI row and the abstain verdict."""
    if n_claims == 0:
        return {c: 0.0 for c in FEATURE_COLUMNS}
    return {
        "claim_max_contradicts": round(NEUTRAL, 6),
        "claim_mean_contradicts": round(NEUTRAL, 6),
        "claim_contradicted_ratio": 0.0,
        "claim_supported_ratio": 0.0,
        "claim_unsupported_ratio": 1.0,
        "claim_min_entails": round(NEUTRAL, 6),
        "claim_mean_entails": round(NEUTRAL, 6),
        "n_claims": float(n_claims),
    }


def _pairs_for_sample(
    question: str, context: str, answer: str, top_k: int = MAX_CONTEXT_SENTENCES
) -> tuple[list, list]:
    """Build (sentence, claim) NLI pairs plus (claim_index, sentence_index) meta.

    Long contexts (RAGTruth summaries) keep the top-k sentences by lexical
    overlap with each claim, so the evidence search stays local and bounded.
    Short contexts use every sentence.
    """
    claims = split_claims(answer or "")
    sentences = split_sentences(context or "")
    if not claims or not sentences:
        return [], []
    pairs, meta = [], []
    if len(sentences) <= top_k:
        for ci, claim in enumerate(claims):
            for si, sentence in enumerate(sentences):
                pairs.append((sentence, claim))
                meta.append((ci, si))
        return pairs, meta

    sentence_tokens = [_tokens(s) for s in sentences]
    for ci, claim in enumerate(claims):
        claim_tokens = _tokens(claim)
        scored = []
        for si, tokens in enumerate(sentence_tokens):
            union = len(claim_tokens | tokens) or 1
            scored.append((len(claim_tokens & tokens) / union, si))
        picked = sorted(si for _, si in sorted(scored, key=lambda t: (-t[0], t[1]))[:top_k])
        for si in picked:
            pairs.append((sentences[si], claim))
            meta.append((ci, si))
    return pairs, meta


def _aggregate(probs: np.ndarray, meta: list, n_claims: int) -> dict:
    """Reduce pair probabilities into the eight claim-level features."""
    best_ent = np.zeros(n_claims)
    best_con = np.zeros(n_claims)
    for k, (ci, _si) in enumerate(meta):
        con, ent = float(probs[k, LABELS.index("contradiction")]), float(probs[k, LABELS.index("entailment")])
        if ent > best_ent[ci]:
            best_ent[ci] = ent
        if con > best_con[ci]:
            best_con[ci] = con

    contradicted = best_con >= CONTRA_CUTOFF
    supported = (~contradicted) & (best_ent >= ENTAIL_CUTOFF)
    unsupported = ~contradicted & ~supported

    return {
        "claim_max_contradicts": round(float(best_con.max()), 6),
        "claim_mean_contradicts": round(float(best_con.mean()), 6),
        "claim_contradicted_ratio": round(float(contradicted.mean()), 6),
        "claim_supported_ratio": round(float(supported.mean()), 6),
        "claim_unsupported_ratio": round(float(unsupported.mean()), 6),
        "claim_min_entails": round(float(best_ent.min()), 6),
        "claim_mean_entails": round(float(best_ent.mean()), 6),
        "n_claims": float(n_claims),
    }


def compute_claim_features(answer: str, context: str, model, batch_size: int = 128) -> dict:
    """Single-sample API (mirrors extract_all_features_single)."""
    pairs, meta = _pairs_for_sample("", context, answer)
    claims = split_claims(answer or "")
    if not pairs:
        return _empty_row(len(claims) if context.strip() else 0)
    probs = np.asarray(model.predict(pairs, batch_size=batch_size, apply_softmax=True))
    return _aggregate(probs, meta, len(claims))


def extract_claim_features_df(
    df: pd.DataFrame,
    model,
    partial_path: Path | None = None,
    checkpoint_every: int = 500,
    batch_size: int = 128,
    chunk_samples: int = 24,
) -> pd.DataFrame:
    """Chunked, resumable extraction over a frame with sample_id/question/context/answer."""
    done = pd.DataFrame(columns=["sample_id", *FEATURE_COLUMNS])
    if partial_path is not None and partial_path.exists():
        done = pd.read_parquet(partial_path)
        df = df[~df["sample_id"].isin(set(done["sample_id"]))]
        logger.info(f"Resumed: {len(done)} samples already done, {len(df)} remaining")

    results: list = []
    pending: list = []
    total = len(df)
    t0 = time.time()
    processed = 0
    last_log = 0
    next_checkpoint = checkpoint_every

    def flush_rows(rows: list) -> None:
        nonlocal results
        if not rows:
            return
        results.extend(rows)
        if partial_path is not None and processed >= next_checkpoint:
            frame = pd.DataFrame(rows)
            combined = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
            combined.to_parquet(partial_path, index=False)

    rows_iter = df.to_dict("records")
    for start in range(0, total, chunk_samples):
        chunk = rows_iter[start : start + chunk_samples]
        pairs: list = []
        meta: list = []
        spans: list = []
        for record in chunk:
            sample_pairs, sample_meta = _pairs_for_sample(
                str(record.get("question", "")), str(record.get("context", "")), str(record.get("answer", ""))
            )
            n_claims = len(split_claims(str(record.get("answer", ""))))
            offset = len(pairs)
            pairs.extend(sample_pairs)
            meta.extend(sample_meta)
            spans.append((record["sample_id"], offset, len(sample_pairs), n_claims, bool(str(record.get("context", "")).strip())))

        probs = np.asarray(model.predict(pairs, batch_size=batch_size, apply_softmax=True)) if pairs else np.zeros((0, 3))
        for sample_id, offset, count, n_claims, has_context in spans:
            if count == 0:
                row = _empty_row(n_claims if has_context else 0)
            else:
                row = _aggregate(probs[offset : offset + count], meta[offset : offset + count], n_claims)
            pending.append({"sample_id": sample_id, **row})

        processed += len(chunk)
        if processed >= next_checkpoint:
            flush_rows(pending)
            pending = []
            next_checkpoint += checkpoint_every
        if processed - last_log >= 1000 or processed == total:
            last_log = processed
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = total - processed
            logger.info(
                f"Claim features: {processed}/{total} ({100 * processed / total:.1f}%) | "
                f"{rate:.1f} samples/s | ETA {remaining / rate / 60:.1f} min"
                if rate > 0
                else f"Claim features: {processed}/{total} samples"
            )

    if pending:
        results.extend(pending)
    out = pd.DataFrame(results)
    if not done.empty:
        out = pd.concat([done, out], ignore_index=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Claim-level NLI aggregation features (b6 M2)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N rows")
    parser.add_argument("--device", default=os.environ.get("HALU_CLAIM_DEVICE", "cuda"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-samples", type=int, default=24)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--fresh", action="store_true", help="ignore an existing partial checkpoint")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found")
    df = pd.read_parquet(args.input)[["sample_id", "question", "context", "answer"]]
    if args.limit:
        df = df.head(args.limit)
    logger.info(f"Input: {args.input} ({len(df)} samples)")

    partial = args.output.with_suffix(".partial.parquet")
    if args.fresh and partial.exists():
        partial.unlink()

    model_kwargs = None
    try:
        import torch

        if args.device == "cuda" and torch.cuda.is_available() and os.environ.get("HALU_CLAIM_FP16", "1") == "1":
            model_kwargs = {"torch_dtype": torch.float16}
            logger.info("CUDA fp16 enabled for the NLI encoder")
    except ImportError:
        pass
    model, chosen = load_nli_model(device=args.device, model_kwargs=model_kwargs)
    logger.info(f"NLI model: {chosen}")

    out = extract_claim_features_df(
        df,
        model,
        partial_path=partial,
        checkpoint_every=args.checkpoint_every,
        batch_size=args.batch_size,
        chunk_samples=args.chunk_samples,
    )
    missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = out[["sample_id", *FEATURE_COLUMNS]].sort_values("sample_id").reset_index(drop=True)
    out.to_parquet(args.output, index=False)
    if partial.exists():
        partial.unlink()
    logger.info(f"Saved {len(out)} rows x {len(FEATURE_COLUMNS)} features -> {args.output}")
    print(out[FEATURE_COLUMNS].describe().round(4).to_string())


if __name__ == "__main__":
    main()
