"""
Pure label-mapping functions for the B1 unified schema (roadmap §14 B1.1/B1.4).

Every mapping is a deterministic, unit-tested function. The binary `label`
is a documented interface; original taxonomy (RAGTruth spans, FaithBench
annotation labels) is preserved losslessly in `span_annotations`.

Mappings:
  - HaluEval:    correct answer -> 0, hallucinated answer -> 1 (fixed at build).
  - RAGTruth:    1 if the response carries any hallucination span, else 0
                 (official spans are the human annotation of hallucination).
  - FaithBench:  severity aggregation over annotation labels:
                   no annotations -> 0
                   Benign -> 0
                   Questionable -> 1 (primary mapping)
                   Unwanted / Unwanted.Intrinsic / Unwanted.Extrinsic -> 1
                 Primary aggregation = worst severity (official script default).
"""

RAGTRUTH_LABEL_MAPPING = "ragtruth-span-v1"
FAITHBENCH_LABEL_MAPPING_PRIMARY = "faithbench-worst-q+unwanted-v1"

# Severity used by the official FaithBench binarize.py (1 = least, 3 = most severe)
FAITHBENCH_SEVERITY = {
    "Benign": 1,
    "Questionable": 2,
    "Unwanted": 3,
    "Unwanted.Intrinsic": 3,
    "Unwanted.Extrinsic": 3,
}

# The official README example contains a typo ("Instrinsic"); the schema and
# released data use "Intrinsic". Normalize so both spellings map identically.
_FAITHBENCH_TYPO_MAP = {"Unwanted.Instrinsic": "Unwanted.Intrinsic"}

FAITHBENCH_PRIMARY_CLASSES = frozenset(
    {"Questionable", "Unwanted", "Unwanted.Intrinsic", "Unwanted.Extrinsic"}
)
FAITHBENCH_STRICT_CLASSES = frozenset({"Unwanted", "Unwanted.Intrinsic", "Unwanted.Extrinsic"})

# Sensitivity configurations reported in the mapping report (B1.6)
FAITHBENCH_SENSITIVITY_CONFIGS = {
    "primary_worst_q_plus_unwanted": {
        "aggregation": "worst",
        "hallucinated_classes": FAITHBENCH_PRIMARY_CLASSES,
    },
    "majority_q_plus_unwanted": {
        "aggregation": "majority",
        "hallucinated_classes": FAITHBENCH_PRIMARY_CLASSES,
    },
    "strict_worst_unwanted_only": {
        "aggregation": "worst",
        "hallucinated_classes": FAITHBENCH_STRICT_CLASSES,
    },
}


def normalize_faithbench_label(label: str) -> str:
    return _FAITHBENCH_TYPO_MAP.get(label, label)


def ragtruth_label_from_spans(spans) -> int:
    """Any human-annotated hallucination span => 1; empty annotation list => 0."""
    return 1 if spans else 0


def faithbench_annotation_labels(annotations) -> set:
    """Set of normalized label strings across all annotations of a sample."""
    labels = set()
    for ann in annotations or []:
        for lab in ann.get("label") or []:
            labels.add(normalize_faithbench_label(lab))
    return labels


def faithbench_severity(label: str) -> int:
    normalized = normalize_faithbench_label(label)
    if normalized not in FAITHBENCH_SEVERITY:
        raise ValueError(f"unknown FaithBench label: {label!r}")
    return FAITHBENCH_SEVERITY[normalized]


def faithbench_label(
    annotations,
    aggregation: str = "worst",
    hallucinated_classes=FAITHBENCH_PRIMARY_CLASSES,
) -> int:
    """Aggregate annotation labels to a binary label.

    aggregation="worst": most severe label wins (official script default).
    aggregation="majority": most frequent severity wins (ties -> most severe).
    """
    labels = faithbench_annotation_labels(annotations)
    if not labels:
        return 0
    severities = [faithbench_severity(l) for l in labels]
    if aggregation == "worst":
        chosen_sev = max(severities)
    elif aggregation == "majority":
        chosen_sev = max(set(severities), key=severities.count)
    else:
        raise ValueError(f"unknown aggregation strategy: {aggregation!r}")
    chosen = next(l for l in set(labels) if faithbench_severity(l) == chosen_sev)
    return 1 if chosen in hallucinated_classes else 0


def faithbench_label_sensitivity(annotations) -> dict:
    """All configured FaithBench labelings for the sensitivity report."""
    return {
        name: int(faithbench_label(annotations, **cfg))
        for name, cfg in FAITHBENCH_SENSITIVITY_CONFIGS.items()
    }
