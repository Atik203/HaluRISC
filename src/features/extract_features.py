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

if __name__ == "__main__":
    parquet_path = os.path.join("data", "processed", "qa_clean.parquet")
    output_path = os.path.join("data", "processed", "features_core.parquet")

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"{parquet_path} not found. Run src/data/prepare.py first.")

    df = pd.read_parquet(parquet_path)
    features_df = extract_all_core_features(df)
    features_df.to_parquet(output_path, index=False)
    logging.info(f"Saved core feature matrix to {output_path}")
