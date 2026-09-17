"""B3 cross-domain tests (synthetic fixtures; no heavy models, no downloads)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.run_b3_cross_domain import (  # noqa: E402
    MODEL_THRESHOLD,
    aggregate_metrics,
    faithbench_sensitivity,
    group_bootstrap_cis,
    predict_zero_shot,
    sample_error_cases,
    select_datasets,
    span_type_subgroups,
    subgroup_metrics,
    word_bin,
)


def make_unified(n_groups: int = 30, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_groups):
        for j, (task, split) in enumerate((("qa", "test"), ("summarization", "train"))):
            label = int(rng.random() < 0.5)
            rows.append({
                "sample_id": f"ragtruth:{i}_{j}",
                "source_dataset": "ragtruth",
                "source_group_id": f"ragtruth:{i}",
                "task": task,
                "domain": "marco" if task == "qa" else "cnn_dm",
                "question": f"q {i}",
                "context": "context " * (40 + i),
                "answer": "answer " * (8 + j),
                "label": label,
                "span_annotations": "[]" if label == 0 else json.dumps([{"label_type": "Evident Conflict", "start": 0, "end": 3}]),
                "generator_model": "gpt-3.5-turbo-0613",
                "official_split": split,
                "experiment_split": "",
                "quality": "good",
                "native_record_id": str(i),
                "native_metadata": "{}",
                "label_mapping_version": "b1-labels-v1",
            })
    rows += [
        {
            "sample_id": f"fb:{k}",
            "source_dataset": "faithbench",
            "source_group_id": f"faithbench:raw_{k}",
            "task": "summarization",
            "domain": "faithbench",
            "question": "",
            "context": "source text " * 30,
            "answer": "summary " * 10,
            "label": 1 if k % 2 else 0,
            "span_annotations": json.dumps([{"label": ["Questionable"]}]) if k % 2 else "[]",
            "generator_model": f"model-{k % 3}",
            "official_split": "",
            "experiment_split": "",
            "quality": "",
            "native_record_id": str(k),
            "native_metadata": "{}",
            "label_mapping_version": "b1-labels-v1",
        }
        for k in range(20)
    ]
    return pd.DataFrame(rows)


class DummyModel:
    """Deterministic pseudo-classifier with predict_proba for threshold tests."""

    def __init__(self, rng_seed: int = 0):
        self.rng = np.random.default_rng(rng_seed)

    def predict_proba(self, X):
        p = self.rng.random(len(X))
        return np.stack([1 - p, p], axis=1)


def test_fixed_threshold_is_05():
    assert MODEL_THRESHOLD == 0.5


def test_select_datasets_filters_correctly():
    df = make_unified()
    subsets = select_datasets(df)
    qa_test = subsets["ragtruth_qa_test"]
    assert (qa_test["task"] == "qa").all()
    assert (qa_test["official_split"] == "test").all()
    assert set(subsets["faithbench"]["source_dataset"]) == {"faithbench"}
    assert set(subsets["ragtruth_all"]["source_dataset"]) == {"ragtruth"}
    for name in ("qa", "summarization"):
        assert (subsets[f"ragtruth_task_{name}"]["task"] == name).all()


def test_predict_uses_fixed_threshold():
    df = make_unified(n_groups=12)
    feats = pd.DataFrame({"f1": np.arange(len(df)) / len(df), "f2": 0.5}, index=df.index)
    df = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    model = DummyModel()
    proba, preds = predict_zero_shot(model, df, ["f1", "f2"])
    assert (preds == (proba >= MODEL_THRESHOLD).astype(int)).all()
    assert set(preds) <= {0, 1}


def test_group_bootstrap_deterministic_and_bounded():
    rng = np.random.default_rng(7)
    df = make_unified(n_groups=15)
    groups = df["source_group_id"].values
    proba = rng.random(len(df))
    y = df["label"].values
    ci1 = group_bootstrap_cis(y, proba, groups, n=50, seed=777)
    ci2 = group_bootstrap_cis(y, proba, groups, n=50, seed=777)
    assert ci1 == ci2
    assert ci1["groups"] == len(np.unique(groups))
    for ci in (ci1["f1_ci"], ci1["auroc_ci"]):
        if ci is not None:
            assert 0.0 <= ci[0] <= ci[1] <= 1.0


def test_subgroup_minimum_rules():
    df = make_unified(n_groups=30)
    rng = np.random.default_rng(3)
    proba = rng.random(len(df))
    preds = (proba >= 0.5).astype(int)
    rows = subgroup_metrics(df, proba, preds, "task")
    by_name = {r["subgroup"]: r for r in rows}
    # task groups have 30 rows each but only 30 groups... n_groups check per subgroup
    for r in rows:
        assert "reported" in r
    # with tiny data some subgroups must be below minimums and flagged
    small = make_unified(n_groups=2)
    proba_s = rng.random(len(small))
    rows_s = subgroup_metrics(small, proba_s, (proba_s >= 0.5).astype(int), "task")
    assert all(r["reported"] is False for r in rows_s)


def test_span_type_subgroups_overlapping():
    df = make_unified(n_groups=40)
    rng = np.random.default_rng(1)
    proba = rng.random(len(df))
    preds = (proba >= 0.5).astype(int)
    rows = span_type_subgroups(df, proba, preds)
    assert all(r["dimension"] == "label_type" for r in rows)
    # a hallucinated row with an Evident Conflict span must appear in that group's mask
    rows_by_type = {r["subgroup"]: r for r in rows if r["reported"]}
    if "Evident Conflict" in rows_by_type:
        assert rows_by_type["Evident Conflict"]["n_rows"] > 0


def test_faithbench_sensitivity_changes_labels_not_predictions():
    df = make_unified()
    fb = df[df["source_dataset"] == "faithbench"].reset_index(drop=True)
    rng = np.random.default_rng(9)
    proba = rng.random(len(fb))
    sens = faithbench_sensitivity(fb, proba)
    assert set(sens) == {"primary_worst_q_plus_unwanted", "majority_q_plus_unwanted", "strict_worst_unwanted_only"}
    primary_pos = sens["primary_worst_q_plus_unwanted"]["n_positive"]
    strict_pos = sens["strict_worst_unwanted_only"]["n_positive"]
    assert strict_pos <= primary_pos  # Questionable-only rows drop out under the strict mapping


def test_word_bin():
    assert word_bin("w " * 50, [("lt_128", 0, 128), ("128_511", 128, 512), ("512_1023", 512, 1024), ("ge_1024", 1024, None)]) == "lt_128"
    assert word_bin("w " * 600, [("lt_128", 0, 128), ("128_511", 128, 512), ("512_1023", 512, 1024), ("ge_1024", 1024, None)]) == "512_1023"
    assert word_bin("w " * 5000, [("lt_128", 0, 128), ("128_511", 128, 512), ("512_1023", 512, 1024), ("ge_1024", 1024, None)]) == "ge_1024"


def test_aggregate_metrics_schema():
    y = np.array([0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    preds = (proba >= 0.5).astype(int)
    m = aggregate_metrics(y, proba, preds)
    for k in ("precision", "recall", "f1", "auroc", "pr_auc", "mcc", "ece", "confusion"):
        assert k in m
    assert m["confusion"]["tn"] + m["confusion"]["fp"] + m["confusion"]["fn"] + m["confusion"]["tp"] == len(y)


def test_error_case_sampling_schema():
    df = make_unified(n_groups=25)
    rng = np.random.default_rng(2)
    proba = rng.random(len(df))
    preds = (proba >= 0.5).astype(int)
    cases = sample_error_cases(df, proba, preds, cap=5)
    assert len(cases) > 0
    assert {"sample_id", "source_dataset", "task", "label", "prediction", "raw_score",
            "source_group_id", "span_annotations"} <= set(cases[0])
    assert set(c["group"] for c in cases) <= {"false_positive", "false_negative",
                                              "high_conf_correct", "high_conf_incorrect"}


def test_error_cases_redact_faithbench_text():
    df = make_unified(n_groups=25)
    fb = df[df["source_dataset"] == "faithbench"].reset_index(drop=True)
    rng = np.random.default_rng(2)
    proba = rng.random(len(fb))
    preds = (proba >= 0.5).astype(int)
    cases = sample_error_cases(fb, proba, preds, cap=20)
    assert len(cases) > 0
    for c in cases:
        assert c["source_dataset"] == "faithbench"
        assert c["context"] == "" and c["answer"] == "" and c["question"] == ""
        assert c.get("text_redacted", "").startswith("FaithBench is CC BY-NC-SA")
