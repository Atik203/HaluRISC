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
# NER only: tagger/parser/lemmatizer/attribute_ruler do not affect .ents and
# cost most of the runtime on long documents.
DISABLE_COMPONENTS = ["tagger", "parser", "attribute_ruler", "lemmatizer"]


def load_ner_model():
    """Load the spaCy NER pipeline (lazy, cached at call site)."""
    import spacy

    try:
        return spacy.load(MODEL_NAME, disable=DISABLE_COMPONENTS)
    except ValueError:
        return spacy.load(MODEL_NAME)


def _entity_features_from_sets(ans_entities: set, ctx_entities: set) -> dict:
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


def extract_entity_features(question: str, context: str, answer: str, nlp) -> dict:
    """Group 3: named-entity overlap between answer and context."""
    ans_entities = {e.text.lower() for e in nlp(answer).ents}
    ctx_entities = {e.text.lower() for e in nlp(context).ents} if context.strip() else set()
    return _entity_features_from_sets(ans_entities, ctx_entities)


def extract_entity_features_df(
    df: pd.DataFrame, nlp, batch_size: int = 64, log_every: int = 500
) -> pd.DataFrame:
    """Batch entity features with spaCy ``nlp.pipe``.

    Answers and contexts are streamed through one pipe (2 documents per row).
    ``HALU_SPACY_N_PROCESS`` parallelizes across CPU cores (Windows-safe under
    the script __main__ guard). Progress is logged with percent and ETA so a
    long run is visible while it works.
    """
    import os
    import time

    total = len(df)
    logger.info(f"Extracting entity (NER) features for {total} samples...")
    texts = []
    for _, row in df.iterrows():
        context = str(row["context"]) if str(row["context"]).strip() else ""
        texts.append(str(row["answer"]))
        texts.append(context)

    n_process = max(1, int(os.environ.get("HALU_SPACY_N_PROCESS", "1")))
    logger.info(f"NER pipe: {len(texts)} docs, batch_size={batch_size}, n_process={n_process}")

    rows = []
    pending_answer: set | None = None
    t0 = time.time()
    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size, n_process=n_process)):
        entities = {e.text.lower() for e in doc.ents}
        if i % 2 == 0:
            pending_answer = entities
            continue
        rows.append(_entity_features_from_sets(pending_answer or set(), entities))
        done = len(rows)
        if log_every and done % log_every == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else 0.0
            logger.info(
                f"Entity features: {done}/{total} ({100 * done / total:.1f}%) | {rate:.1f} samples/s | ETA {eta / 60:.1f} min"
            )
    logger.info(f"Entity features done: {len(rows)} rows in {time.time() - t0:.1f}s")
    return pd.DataFrame(rows, index=df.index)
