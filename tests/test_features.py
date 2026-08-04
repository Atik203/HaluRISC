"""Unit tests for HaluRISC feature extraction (all 7 groups)."""

import numpy as np
import pytest

from src.features.extract_features import (
    extract_hedging_features,
    extract_length_features,
    extract_lexical_features,
    extract_numeric_features,
)


# --------------------------------------------------------------------------
# Group 1: length / style
# --------------------------------------------------------------------------
def test_length_features_basic():
    f = extract_length_features("q", "c", "Hello world. Second sentence!")
    assert f["n_words"] == 4
    assert f["n_sentences"] == 2
    assert f["n_chars"] == len("Hello world. Second sentence!")
    assert f["avg_word_len"] > 0


def test_length_features_empty_answer():
    f = extract_length_features("q", "c", "")
    assert f["n_words"] == 0
    assert f["n_sentences"] == 1  # guarded: at least 1 sentence
    assert f["avg_word_len"] == 0


# --------------------------------------------------------------------------
# Group 2: lexical overlap
# --------------------------------------------------------------------------
def test_lexical_overlap_perfect_grounding():
    f = extract_lexical_features("What is the capital?", "Paris is the capital of France.", "Paris is the capital of France.")
    assert f["overlap_answer_context"] == pytest.approx(1.0, abs=1e-6)


def test_lexical_overlap_empty_context():
    f = extract_lexical_features("q", "", "Some answer text here.")
    assert f["overlap_answer_context"] == 0.0
    assert f["jaccard_ans_ctx"] == 0.0


# --------------------------------------------------------------------------
# Group 5: numeric consistency
# --------------------------------------------------------------------------
def test_numeric_features_novel_numbers():
    f = extract_numeric_features("When?", "Discovered in 1928.", "Discovered in 1945 and 2020.")
    assert f["novel_numbers"] == 2
    assert f["number_overlap_ratio"] == pytest.approx(0.0)


def test_numeric_features_no_numbers_in_answer():
    f = extract_numeric_features("q", "In 1928.", "No numbers here")
    assert f["novel_numbers"] == 0
    assert f["number_overlap_ratio"] == 1.0  # no numbers -> no numeric hallucination signal


# --------------------------------------------------------------------------
# Group 6: hedging
# --------------------------------------------------------------------------
def test_hedging_count():
    f = extract_hedging_features("q", "c", "It might be true and probably is.")
    assert f["hedge_count"] == 2
    assert f["hedge_density"] == pytest.approx(2 / 7, abs=1e-6)


# --------------------------------------------------------------------------
# Group 3/4/7: entity, NLI, semantic (model-backed; neutral/empty handling)
# --------------------------------------------------------------------------
def test_nli_neutral_when_empty_context():
    from src.features.nli_features import extract_nli_features

    f = extract_nli_features("q", "", "An answer.", model=None)
    assert f["nli_ctx_entails_ans"] == pytest.approx(1 / 3)
    assert f["nli_ctx_contradicts_ans"] == pytest.approx(1 / 3)
    assert f["nli_ctx_neutral_ans"] == pytest.approx(1 / 3)


def test_semantic_empty_context_is_zero():
    from src.features.semantic_features import extract_semantic_features

    class StubEncoder:
        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 384))

    f = extract_semantic_features("question", "", "answer", StubEncoder())
    assert f["cosine_ctx_ans"] == 0.0
    assert f["cosine_q_ans"] == 0.0


@pytest.mark.integration
def test_entity_features_no_entities():
    from src.features.entity_features import extract_entity_features

    class StubNlp:
        class Doc:
            ents = []

        def __call__(self, text):
            return self.Doc()

    f = extract_entity_features("q", "context without entities", "plain answer without entities", StubNlp())
    assert f["entity_overlap_ratio"] == 1.0
    assert f["novel_entity_ratio"] == 0.0
