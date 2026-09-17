"""B2 baseline/control tests (synthetic fixtures; no downloads, no heavy models)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.run_b2_baselines import (  # noqa: E402
    B2Config,
    MODEL_THRESHOLD,
    SMOKE_GRID,
    build_synthetic,
    check_group_cv_disjoint,
    heuristic_overlap,
    run_experiment,
    run_tuning,
)


def test_group_cv_folds_are_group_disjoint():
    data = build_synthetic(n_groups=80, seed=3)
    features = data["features"]
    train_df = features[features["split"] == "train"]
    groups = train_df["item_idx"].values
    X = np.zeros((len(train_df), 5))
    y = train_df["label"].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    report = check_group_cv_disjoint(cv, X, y, groups)
    assert report["n_splits"] == 5
    assert report["overlapping_groups_across_folds"] == 0
    assert all(f["train_groups"] > 0 and f["val_groups"] > 0 for f in report["groups_per_fold"])


def test_tuning_uses_group_cv():
    data = build_synthetic(n_groups=60, seed=5)
    features = data["features"]
    train_df = features[features["split"] == "train"]
    X = train_df[data["feature_cols"]].values
    y = train_df["label"].values
    groups = train_df["item_idx"].values
    cfg = B2Config(n_iter=2, tuning_grid=SMOKE_GRID, smoke=True)
    best_params, best_score, cv_report = run_tuning(X, y, groups, seed=42, cfg=cfg)
    assert isinstance(best_params, dict)
    assert cv_report["overlapping_groups_across_folds"] == 0
    assert 0 <= best_score <= 1


def test_tfidf_fit_on_train_only():
    train_texts = ["alpha beta gamma", "delta epsilon zeta"]
    test_only = ["uniquetesttokenonly"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(train_texts)
    vocab = set(vec.get_feature_names_out())
    assert "uniquetesttokenonly" not in vocab
    assert "alpha" in vocab
    X_test = vec.transform(test_only)
    assert X_test.sum() == 0  # unseen token contributes nothing


def test_heuristic_threshold_uses_validation_only():
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({
        "label": rng.integers(0, 2, n),
        "overlap_answer_context": rng.random(n),
        "split": "val",
    })
    test_df = df.copy()
    test_df["split"] = "test"
    test_df["overlap_answer_context"] = test_df["overlap_answer_context"] + 0.5  # shifted distribution
    preds, probs, info = heuristic_overlap(df, df, test_df)
    # threshold must come from validation; test shift must not alter it
    assert 0.0 <= info["threshold"] <= 1.0
    assert info["val_f1"] >= 0.0
    # re-running with a different test distribution yields the SAME threshold
    test_df2 = df.copy()
    test_df2["split"] = "test"
    test_df2["overlap_answer_context"] = test_df2["overlap_answer_context"] - 0.5
    _, _, info2 = heuristic_overlap(df, df, test_df2)
    assert info["threshold"] == info2["threshold"]


def test_model_threshold_is_05():
    assert MODEL_THRESHOLD == 0.5


def test_majority_balanced_ties_resolve_to_zero():
    data = build_synthetic(n_groups=40, seed=11)
    cfg = B2Config(results_dir=ROOT / "artifacts" / "results" / "b2_test_tmp",
                   models_dir=ROOT / "artifacts" / "models" / "b2_test_tmp",
                   seeds=[42], n_iter=2, tuning_grid=SMOKE_GRID,
                   tfidf_max_features=500, tfidf_min_df=1, smoke=True)
    out = run_experiment(cfg, data)
    maj = out["comparison"]["majority"]
    assert maj["deterministic"] is True
    assert maj["recall_mean"] == 0.0  # predicts 0 everywhere


def test_smoke_end_to_end_artifacts():
    data = build_synthetic(n_groups=50, seed=9)
    cfg = B2Config(results_dir=ROOT / "artifacts" / "results" / "b2_smoke_test",
                   models_dir=ROOT / "artifacts" / "models" / "b2_smoke_test",
                   seeds=[42], n_iter=2, tuning_grid=SMOKE_GRID,
                   tfidf_max_features=500, tfidf_min_df=1, smoke=True)
    out = run_experiment(cfg, data)
    expected = [
        "b2_model_comparison.csv", "b2_model_comparison.json", "b2_per_seed_metrics.csv",
        "b2_predictions.parquet", "b2_confusion_matrices.json", "b2_tuning.json",
        "b2_statistical_tests.json", "b2_bootstrap_cis.json", "b2_leakage_comparison.json",
        "b2_run_config.json",
    ]
    for name in expected:
        assert (cfg.results_dir / name).exists(), f"missing {name}"
    models = list(cfg.models_dir.glob("*.joblib"))
    assert len(models) >= 4  # xgboost + rf + tfidf x3 + scalers (subset is enough)
    comparison = out["comparison"]
    assert "majority" in comparison and "heuristic_overlap" in comparison
    assert "xgboost" in comparison and "lr_full" in comparison
    preds = pd.read_parquet(cfg.results_dir / "b2_predictions.parquet")
    assert {"sample_id", "item_idx", "label", "split", "model", "seed", "threshold", "score", "pred"} <= set(preds.columns)
    assert (preds["split"] == "test").all()
