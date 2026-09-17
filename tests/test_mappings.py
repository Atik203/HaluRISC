"""Unit tests for B1 label mappings (src/data/mappings.py). No data downloads."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.mappings import (  # noqa: E402
    FAITHBENCH_PRIMARY_CLASSES,
    FAITHBENCH_STRICT_CLASSES,
    faithbench_annotation_labels,
    faithbench_label,
    faithbench_label_sensitivity,
    normalize_faithbench_label,
    ragtruth_label_from_spans,
)


def test_ragtruth_empty_spans_are_faithful():
    assert ragtruth_label_from_spans([]) == 0


def test_ragtruth_any_span_is_hallucinated():
    assert ragtruth_label_from_spans([{"start": 0, "end": 5, "label_type": "Evident Conflict"}]) == 1


def test_faithbench_no_annotations_is_faithful():
    assert faithbench_label([]) == 0


def test_faithbench_benign_is_faithful():
    annotations = [{"label": ["Benign"]}]
    assert faithbench_label(annotations) == 0


def test_faithbench_questionable_is_positive():
    annotations = [{"label": ["Questionable"]}]
    assert faithbench_label(annotations) == 1


def test_faithbench_unwanted_is_positive():
    for label in ("Unwanted", "Unwanted.Intrinsic", "Unwanted.Extrinsic"):
        assert faithbench_label([{"label": [label]}]) == 1


def test_faithbench_typo_normalization():
    assert normalize_faithbench_label("Unwanted.Instrinsic") == "Unwanted.Intrinsic"
    assert faithbench_label([{"label": ["Unwanted.Instrinsic"]}]) == 1
    labels = faithbench_annotation_labels([{"label": ["Unwanted.Instrinsic"]}])
    assert labels == {"Unwanted.Intrinsic"}


def test_faithbench_worst_beats_majority():
    # Benign (sev 1) appears twice, Unwanted (sev 3) once -> worst = 1, majority = 0
    annotations = [{"label": ["Benign"]}, {"label": ["Benign"]}, {"label": ["Unwanted"]}]
    assert faithbench_label(annotations, aggregation="worst") == 1
    assert faithbench_label(annotations, aggregation="majority") == 0


def test_faithbench_unknown_aggregation_raises():
    with pytest.raises(ValueError):
        faithbench_label([{"label": ["Benign"]}], aggregation="average")


def test_faithbench_strict_mapping_excludes_questionable():
    annotations = [{"label": ["Questionable"]}]
    assert faithbench_label(annotations, hallucinated_classes=FAITHBENCH_STRICT_CLASSES) == 0


def test_faithbench_sensitivity_configs_cover_primary():
    assert faithbench_label_sensitivity([{"label": ["Questionable"]}])["primary_worst_q_plus_unwanted"] == 1
    assert faithbench_label_sensitivity([{"label": ["Questionable"]}])["strict_worst_unwanted_only"] == 0
    assert set(FAITHBENCH_PRIMARY_CLASSES) >= FAITHBENCH_STRICT_CLASSES
