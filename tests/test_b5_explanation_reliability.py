"""B5 explanation-reliability tests (fast: text transforms + tiny models, no heavy downloads)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.run_b5_explanation_reliability import (  # noqa: E402
    apply_perturbation,
    bootstrap_shap_stability,
    group_ablation_deltas,
    mean_abs_shap_ranking,
    neutralize_topk,
    perturb_clause_shuffle,
    perturb_date,
    perturb_irrelevant_insert,
    perturb_numeric,
    perturb_support_removal,
    sample_review_cases,
)


def _tiny_model():
    import xgboost

    from src.models.train_pipeline import FEATURE_GROUPS

    cols = FEATURE_GROUPS["length"] + FEATURE_GROUPS["numeric"][:2]  # real names for group ablation
    rng = np.random.default_rng(0)
    X = rng.random((300, len(cols)))
    y = ((X[:, 0] + 0.5 * X[:, 1]) > 1.0).astype(int)  # first features dominate
    m = xgboost.XGBClassifier(n_estimators=10, max_depth=3, device="cpu", tree_method="hist")
    m.fit(X, y)
    return m, X, y, cols


# ---- text perturbations ----

def test_numeric_replacement_changes_numbers():
    out, meta = perturb_numeric("It costs 42 dollars and 3.5 cents.", np.random.default_rng(1))
    assert meta["changed"] and meta["n_replaced"] >= 2
    assert "42 dollars" not in out and "3.5" not in out  # tokens replaced (may embed digits)


def test_numeric_replacement_noop_without_numbers():
    out, meta = perturb_numeric("Nothing numeric here.", np.random.default_rng(1))
    assert meta["changed"] is False and out == "Nothing numeric here."


def test_date_replacement():
    out, meta = perturb_date("The event on March 12, 2021 was cancelled.", np.random.default_rng(2))
    assert meta["changed"] is True
    assert "March 12, 2021" not in out


def test_support_sentence_removal_picks_max_overlap():
    ctx = "Paris is the capital of France. The Louvre museum is large. It rains a lot."
    ans = "Paris is the capital of France."
    out, meta = perturb_support_removal(ctx, ans)
    assert meta["changed"] is True
    assert "capital of France" not in out
    assert "Louvre" in out  # other sentences kept


def test_support_removal_noop_single_sentence():
    out, meta = perturb_support_removal("Only one sentence here.", "answer text")
    assert meta["changed"] is False


def test_irrelevant_insertion():
    out, meta = perturb_irrelevant_insert("Base context.", np.random.default_rng(1))
    assert meta["changed"] is True and len(out) > len("Base context.")


def test_clause_shuffle_deterministic_and_preserves_tokens():
    ans = "First clause; second clause, third clause."
    out1, m1 = perturb_clause_shuffle(ans, np.random.default_rng(5))
    out2, m2 = perturb_clause_shuffle(ans, np.random.default_rng(5))
    assert m1["changed"] and out1 == out2  # deterministic per seed
    assert sorted(out1.replace(",", " ").replace(";", " ").split()) == \
        sorted(ans.replace(",", " ").replace(";", " ").split())


def test_apply_perturbation_all_types_defined():
    for kind in ("numeric", "date", "entity", "support_removal", "irrelevant_insert", "clause_shuffle"):
        q, c, a, meta = apply_perturbation(
            kind, "Q?", "Context with one sentence.", "Answer with 12 numbers.", seed=42, nlp=None)
        assert isinstance(q, str) and isinstance(c, str) and isinstance(a, str)
        assert isinstance(meta, dict)


# ---- importance / neutralization / stability ----

def test_mean_abs_shap_ranking_sorted():
    shap_vals = np.array([[3.0, -1.0], [1.0, 0.5]])
    ranking = mean_abs_shap_ranking(shap_vals, ["a", "b"])
    assert list(ranking.keys()) == ["a", "b"]


def test_neutralize_topk_drops_score_for_dominant_feature():
    import xgboost

    m, X, y, cols = _tiny_model()
    import shap

    sv = np.asarray(shap.TreeExplainer(m).shap_values(X))
    rows = neutralize_topk(m, X, y, sv, cols, ks=(1, 3))
    assert len(rows) == 2
    assert rows[0]["k"] == 1
    assert rows[0]["mean_score_delta"] < -0.05  # neutralizing the dominant feature must move scores


def test_group_ablation_deltas_runs():
    import shap

    m, X, y, cols = _tiny_model()
    sv = np.asarray(shap.TreeExplainer(m).shap_values(X))
    deltas = group_ablation_deltas(m, X, y, cols)
    assert isinstance(deltas, dict) and "length" in deltas


def test_bootstrap_shap_stability_shapes():
    sv = np.random.default_rng(0).random((200, 4))
    out = bootstrap_shap_stability(sv, ["a", "b", "c", "d"], n=50, seed=42, top_k=3)
    assert out["n_bootstrap"] == 50
    assert set(out["feature_mean_abs_shap_ci"]) == {"a", "b", "c", "d"}
    assert 0.0 <= out["topk_set_jaccard"]["mean"] <= 1.0
    for f in out["feature_mean_abs_shap_ci"].values():
        assert f["lo"] <= f["mean"] <= f["hi"]


# ---- review sampling ----

def test_review_case_sampling_balanced_and_borderline():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(400)],
        "raw_score": rng.random(400),
        "calibrated_score": rng.random(400),
        "label": rng.binomial(1, 0.5, 400),
        "question": ["q"] * 400, "context": ["c"] * 400, "answer": ["a"] * 400,
    })
    picks = sample_review_cases(df, df["calibrated_score"].values, n_per_class=10, n_borderline=20, seed=42)
    assert len(picks) >= 20  # at least the borderline bucket
    assert picks["sample_id"].nunique() == len(picks)
