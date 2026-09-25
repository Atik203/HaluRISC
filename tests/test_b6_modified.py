"""B6 modified-XGBoost unit tests: feature transforms, monotone constraints,
strict-threshold selection, and an end-to-end synthetic smoke run."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import run_b6_modified as b6  # noqa: E402


def test_variant_feature_counts():
    assert len(b6.variant_features("m0")) == 26
    assert len(b6.variant_features("m1")) == 26
    assert len(b6.variant_features("m2")) == 34
    assert len(b6.variant_features("m3")) == 35
    assert b6.variant_features("m3")[-1] == b6.SOURCE_COLUMN
    assert b6.variant_features("m2")[-1] in b6.CLAIM_COLUMNS


def test_constraint_string_values():
    names = b6.variant_features("m2")
    text = b6.constraint_string(names)
    assert text is not None and text.startswith("(") and text.endswith(")")
    values = [int(v) for v in text.strip("()").split(",")]
    assert len(values) == len(names)
    by_name = dict(zip(names, values))
    assert by_name["nli_ctx_contradicts_ans"] == 1
    assert by_name["overlap_answer_context"] == -1
    assert by_name["claim_supported_ratio"] == -1
    assert by_name["claim_max_contradicts"] == 1
    assert by_name["n_sentences"] == 0
    assert b6.SOURCE_COLUMN not in by_name
    m3_values = [int(v) for v in b6.constraint_string(b6.variant_features("m3")).strip("()").split(",")]
    assert m3_values[-1] == 0


def test_log_length_transform():
    df = pd.DataFrame({c: [1.0] for c in b6.BASE_COLUMNS})
    df["n_chars"] = [0.0]
    df["n_words"] = [99.0]
    X = b6.feature_matrix(df, b6.BASE_COLUMNS, log_lengths=True)
    assert X[0, b6.BASE_COLUMNS.index("n_chars")] == 0.0
    assert abs(X[0, b6.BASE_COLUMNS.index("n_words")] - np.log1p(99.0)) < 1e-9
    raw = b6.feature_matrix(df, b6.BASE_COLUMNS, log_lengths=False)
    assert raw[0, b6.BASE_COLUMNS.index("n_words")] == 99.0


def test_strict_threshold_caps_fpr():
    rng = np.random.default_rng(0)
    y = np.array([0] * 900 + [1] * 100)
    p = np.concatenate([rng.random(900), rng.random(100)])
    out = b6.strict_threshold(y, p, alpha=0.05)
    assert out["val_fpr"] <= 0.05
    assert 0.0 < out["threshold"] <= 1.0
    assert 0.0 <= out["val_recall"] <= 1.0


def test_in_domain_and_external_metrics_shapes():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = b6.in_domain_metrics(y, p, 0.5)
    assert metrics["f1"] == 1.0
    assert metrics["confusion"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    ext = b6.external_metrics(y, p, 0.5)
    assert ext["predicted_positive_rate"] == 0.5
    assert ext["recall"] == 1.0


def test_b6_smoke_run(tmp_path, monkeypatch):
    halueval, external = b6.build_synthetic(n_groups=30)
    cfg = b6.B6Config(
        results_dir=tmp_path / "results",
        models_dir=tmp_path / "models",
        variants=("m0", "m2"),
        seeds=[42],
        n_iter=2,
        tuning_grid={"max_depth": [3], "learning_rate": [0.1], "n_estimators": [30],
                     "subsample": [0.9], "colsample_bytree": [0.9]},
        smoke=True,
    )
    monkeypatch.setattr(b6, "load_halueval", lambda: halueval)
    monkeypatch.setattr(b6, "load_external", lambda: external)
    result = b6.run_b6(cfg)
    assert set(result["comparison"]) == {"m0", "m2"}
    assert result["comparison"]["m2"]["n_features"] == 34
    assert (tmp_path / "results" / "b6_model_comparison.csv").exists()
    assert (tmp_path / "results" / "b6_run_config.json").exists()
    assert (tmp_path / "results" / "b6_feature_names.json").exists()
    assert (tmp_path / "results" / "b6_strict_mode.json").exists()
