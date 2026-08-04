"""
Feature extraction pipeline for HaluRISC.
Computes lightweight linguistic, lexical overlap, numeric consistency, and hedging features.
Prepares modular architecture for NER, Semantic, and NLI features.
"""

import os
import re
import json
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Lexicon of hedging / uncertainty indicators
HEDGE_LEXICON = {
    "maybe", "might", "likely", "possibly", "probably", "could", "seems", 
    "uncertain", "unclear", "allegedly", "reportedly", "presumably", 
    "supposedly", "i think", "i believe", "appears to", "it seems"
}

def extract_length_features(question: str, context: str, answer: str) -> dict:
    """Group 1: Length and stylistic features."""
    words = answer.split()
    n_words = len(words)
    n_chars = len(answer)
    sentences = [s for s in re.split(r'[.!?]+', answer) if s.strip()]
    n_sentences = max(1, len(sentences))
    avg_word_len = n_chars / max(1, n_words)

    return {
        "n_chars": n_chars,
        "n_words": n_words,
        "n_sentences": n_sentences,
        "avg_word_len": avg_word_len,
    }

def extract_lexical_features(question: str, context: str, answer: str) -> dict:
    """Group 2: Lexical overlap and grounding features."""
    ans_tokens = set(re.findall(r'\w+', answer.lower()))
    ctx_tokens = set(re.findall(r'\w+', context.lower()))
    q_tokens = set(re.findall(r'\w+', question.lower()))

    if not ans_tokens:
        return {
            "overlap_answer_context": 0.0,
            "overlap_answer_question": 0.0,
            "jaccard_ans_ctx": 0.0,
            "jaccard_ans_q": 0.0
        }

    ans_ctx_intersect = ans_tokens.intersection(ctx_tokens)
    ans_q_intersect = ans_tokens.intersection(q_tokens)

    overlap_ans_ctx = len(ans_ctx_intersect) / len(ans_tokens)
    overlap_ans_q = len(ans_q_intersect) / len(ans_tokens)

    union_ans_ctx = ans_tokens.union(ctx_tokens)
    jaccard_ans_ctx = len(ans_ctx_intersect) / max(1, len(union_ans_ctx))

    union_ans_q = ans_tokens.union(q_tokens)
    jaccard_ans_q = len(ans_q_intersect) / max(1, len(union_ans_q))

    return {
        "overlap_answer_context": overlap_ans_ctx,
        "overlap_answer_question": overlap_ans_q,
        "jaccard_ans_ctx": jaccard_ans_ctx,
        "jaccard_ans_q": jaccard_ans_q
    }

def extract_numeric_features(question: str, context: str, answer: str) -> dict:
    """Group 5: Numeric consistency features."""
    num_pattern = r'\b\d+(?:\.\d+)?%?\b'
    ans_nums = set(re.findall(num_pattern, answer))
    ctx_nums = set(re.findall(num_pattern, context))

    n_nums_ans = len(ans_nums)
    n_nums_ctx = len(ctx_nums)

    if n_nums_ans == 0:
        return {
            "n_numbers_answer": 0,
            "n_numbers_context": n_nums_ctx,
            "number_overlap_ratio": 1.0,  # No numbers in answer -> no numeric hallucination
            "novel_numbers": 0
        }

    overlap_nums = ans_nums.intersection(ctx_nums)
    overlap_ratio = len(overlap_nums) / n_nums_ans
    novel_nums = len(ans_nums - ctx_nums)

    return {
        "n_numbers_answer": n_nums_ans,
        "n_numbers_context": n_nums_ctx,
        "number_overlap_ratio": overlap_ratio,
        "novel_numbers": novel_nums
    }

def extract_hedging_features(question: str, context: str, answer: str) -> dict:
    """Group 6: Hedging and uncertainty features."""
    ans_lower = answer.lower()
    hedge_count = 0
    for hedge in HEDGE_LEXICON:
        if hedge in ans_lower:
            hedge_count += ans_lower.count(hedge)

    words = ans_lower.split()
    hedge_density = hedge_count / max(1, len(words))

    return {
        "hedge_count": hedge_count,
        "hedge_density": hedge_density
    }

def extract_all_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes all core features for a DataFrame containing question, context, answer."""
    logging.info(f"Extracting core features for {len(df)} samples...")
    feature_rows = []

    for idx, row in df.iterrows():
        q, c, a = row["question"], row["context"], row["answer"]
        feats = {}
        feats.update(extract_length_features(q, c, a))
        feats.update(extract_lexical_features(q, c, a))
        feats.update(extract_numeric_features(q, c, a))
        feats.update(extract_hedging_features(q, c, a))
        feature_rows.append(feats)

    features_df = pd.DataFrame(feature_rows, index=df.index)
    result_df = pd.concat([df[["sample_id", "item_idx", "label", "split"]], features_df], axis=1)
    logging.info(f"Successfully extracted {features_df.shape[1]} core features.")
    return result_df


def load_heavy_models(nli_model_name: str | None = None, device: str | None = None) -> dict:
    """Load NER + NLI + embedding models once (used by batch extraction and API).

    device=None -> library default; pass "cpu" for API stability (no VRAM OOM).
    """
    import time

    from src.features.entity_features import load_ner_model
    from src.features.nli_features import load_nli_model
    from src.features.semantic_features import load_embedding_model

    models = {}
    t0 = time.time()
    models["nlp"] = load_ner_model()
    logging.info(f"NER model loaded in {time.time() - t0:.1f}s")
    t0 = time.time()
    models["nli"], nli_name = load_nli_model(nli_model_name, device=device)
    logging.info(f"NLI model ({nli_name}) loaded in {time.time() - t0:.1f}s")
    t0 = time.time()
    models["embedder"] = load_embedding_model(device=device)
    logging.info(f"Embedding model loaded in {time.time() - t0:.1f}s")
    return models


def extract_all_features_single(question: str, context: str, answer: str, models: dict) -> dict:
    """All 7 feature groups for one sample. Used by the FastAPI inference server."""
    from src.features.entity_features import extract_entity_features
    from src.features.nli_features import extract_nli_features
    from src.features.semantic_features import extract_semantic_features

    feats = {}
    feats.update(extract_length_features(question, context, answer))
    feats.update(extract_lexical_features(question, context, answer))
    feats.update(extract_numeric_features(question, context, answer))
    feats.update(extract_hedging_features(question, context, answer))
    feats.update(extract_entity_features(question, context, answer, models["nlp"]))
    feats.update(extract_nli_features(question, context, answer, models["nli"]))
    feats.update(extract_semantic_features(question, context, answer, models["embedder"]))
    return feats


def extract_full_feature_set(df: pd.DataFrame, models: dict | None = None) -> pd.DataFrame:
    """Extracts all 7 feature groups (core + entity + NLI + semantic) with per-group latency."""
    import time

    from src.features.entity_features import extract_entity_features_df
    from src.features.nli_features import extract_nli_features_df
    from src.features.semantic_features import extract_semantic_features_df

    if models is None:
        models = load_heavy_models()

    result_df = extract_all_core_features(df)

    t0 = time.time()
    entity_df = extract_entity_features_df(df, models["nlp"])
    logging.info(f"Entity features done in {time.time() - t0:.1f}s")

    t0 = time.time()
    nli_df = extract_nli_features_df(df, models["nli"])
    logging.info(f"NLI features done in {time.time() - t0:.1f}s")

    t0 = time.time()
    semantic_df = extract_semantic_features_df(df, models["embedder"])
    logging.info(f"Semantic features done in {time.time() - t0:.1f}s")

    result_df = pd.concat([result_df, entity_df, nli_df, semantic_df], axis=1)
    logging.info(f"Full feature matrix: {result_df.shape[1]} features total.")
    return result_df


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    parser = argparse.ArgumentParser(description="HaluRISC full feature extraction")
    parser.add_argument("--input", default=os.path.join("data", "processed", "qa_clean.parquet"))
    parser.add_argument("--output", default=os.path.join("data", "processed", "features_full.parquet"))
    parser.add_argument("--nli-model", default=None, help="Override NLI CrossEncoder checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"{args.input} not found. Run src/data/prepare.py first.")

    df = pd.read_parquet(args.input)
    models = load_heavy_models(args.nli_model)
    features_df = extract_full_feature_set(df, models)
    features_df.to_parquet(args.output, index=False)
    logging.info(f"Saved full feature matrix to {args.output}")
