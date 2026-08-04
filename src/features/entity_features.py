"""
Entity (NER) feature extraction for HaluRISC.

Group 3 features (roadmap §6):
  n_entities_answer, n_entities_context, entity_overlap_ratio, novel_entity_ratio

Uses spaCy `en_core_web_sm` for named entity recognition.
Empty context -> answer entities are all novel (overlap 0). No entities in
answer -> no unsupported-entity signal (overlap 1.0, novel 0.0).
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME = "en_core_web_sm"


def load_ner_model():
    """Load the spaCy NER pipeline (lazy, cached at call site)."""
    import spacy

    return spacy.load(MODEL_NAME)


def extract_entity_features(question: str, context: str, answer: str, nlp) -> dict:
    """Group 3: named-entity overlap between answer and context."""
    ans_entities = {e.text.lower() for e in nlp(answer).ents}
    ctx_entities = {e.text.lower() for e in nlp(context).ents} if context.strip() else set()

    n_ans = len(ans_entities)
    n_ctx = len(ctx_entities)

    if n_ans == 0:
        entity_overlap_ratio = 1.0
        novel_entity_ratio = 0.0
    else:
        inter = ans_entities.intersection(ctx_entities)
        entity_overlap_ratio = len(inter) / n_ans
        novel_entity_ratio = (n_ans - len(inter)) / n_ans

    return {
        "n_entities_answer": n_ans,
        "n_entities_context": n_ctx,
        "entity_overlap_ratio": round(float(entity_overlap_ratio), 6),
        "novel_entity_ratio": round(float(novel_entity_ratio), 6),
    }


def extract_entity_features_df(df: pd.DataFrame, nlp, batch_size: int = 64) -> pd.DataFrame:
    """Batch entity features for a DataFrame with question/context/answer columns."""
    logger.info(f"Extracting entity (NER) features for {len(df)} samples...")
    rows = []
    for _, row in df.iterrows():
        rows.append(extract_entity_features(row["question"], row["context"], row["answer"], nlp))
    return pd.DataFrame(rows, index=df.index)
