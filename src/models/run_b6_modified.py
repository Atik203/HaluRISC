"""
B6 — EC-XGB (Evidence-Consistent XGBoost) against the standard B2 XGBoost,
under the project protocol.

EC-XGB keeps the standard XGBoost learner and changes three things, added
cumulatively in the variants:

  m0  standard XGBoost on the 26 base features (reference, mirrors B2)
  m1  m0 + log-scaled length features + domain monotone constraints
  m2  m1 + 8 claim-level NLI aggregation features (34 total)
  m3  m2 + RAGTruth non-QA train rows and a source indicator (35 total)

The paper name for the full variant (m3) is EC-XGB, Evidence-Consistent
XGBoost. The m0/m1/m2 rows are the component ablations that show which part
earns the name.

Protocol: grouped 5-fold randomized tuning per seed (30 iterations), seeds
42/123/456, early stopping on the HaluEval validation split, raw probabilities
for every metric. In-domain metrics plus zero-shot external evaluation on
RAGTruth QA test, all RAGTruth test tasks, and FaithBench. A strict mode picks
the validation threshold that caps the false-positive rate at alpha.

Run (repo root, .venv):
  python src/models/run_b6_modified.py
  python src/models/run_b6_modified.py --variants m0,m2 --seeds 42
  python src/models/run_b6_modified.py --smoke-test

Outputs: artifacts/results/b6/ and artifacts/models/b6/.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from statsmodels.stats.contingency_tables import mcnemar
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("b6_modified")

from src.features.claim_features import FEATURE_COLUMNS as CLAIM_COLUMNS  # noqa: E402
from src.models.config import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, SEEDS  # noqa: E402
from src.models.train_pipeline import (  # noqa: E402
    FEATURE_GROUPS,
    TUNING_GRID,
    bootstrap_ci,
    ece,
    xgb_device,
)

MODEL_NAME = "EC-XGB"
MODEL_FULL_NAME = "Evidence-Consistent XGBoost"

BASE_COLUMNS = [c for cols in FEATURE_GROUPS.values() for c in cols]
FEATURES_FULL = DATA_PROCESSED / "features_full.parquet"
CLAIM_FEATURES = DATA_PROCESSED / "claim_features.parquet"
EXTERNAL_FEATURES = DATA_PROCESSED / "external_features.parquet"
SOURCE_COLUMN = "source_ragtruth"

LOG_LENGTH_COLUMNS = ("n_chars", "n_words")
DEFAULT_VARIANTS = ("m0", "m1", "m2", "m3")
DEFAULT_ALPHA = 0.05

# Domain priors: risk must not fall when contradiction/novelty rises, and must
# not climb when entailment/overlap rises.
CONSTRAINTS = {
    "overlap_answer_context": -1,
    "jaccard_ans_ctx": -1,
    "nli_ctx_entails_ans": -1,
    "cosine_ctx_ans": -1,
    "nli_ctx_contradicts_ans": 1,
    "novel_entity_ratio": 1,
    "novel_numbers": 1,
    "claim_max_contradicts": 1,
    "claim_mean_contradicts": 1,
    "claim_contradicted_ratio": 1,
    "claim_min_entails": -1,
    "claim_mean_entails": -1,
    "claim_supported_ratio": -1,
}


@dataclass
class B6Config:
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR / "b6")
    models_dir: Path = field(default_factory=lambda: MODELS_DIR / "b6")
    variants: tuple = DEFAULT_VARIANTS
    seeds: list = field(default_factory=lambda: list(SEEDS))
    n_iter: int = 30
    tuning_grid: dict = field(default_factory=lambda: dict(TUNING_GRID))
    alpha: float = DEFAULT_ALPHA
    smoke: bool = False


def variant_features(variant: str) -> list:
    """Ordered feature names for one variant."""
    names = list(BASE_COLUMNS)
    if variant in ("m2", "m3"):
        names += list(CLAIM_COLUMNS)
    if variant == "m3":
        names.append(SOURCE_COLUMN)
    return names


def constraint_string(names: list) -> str | None:
    if not any(name in CONSTRAINTS for name in names):
        return None
    return "(" + ",".join(str(CONSTRAINTS.get(name, 0)) for name in names) + ")"


def feature_matrix(df: pd.DataFrame, names: list, log_lengths: bool) -> np.ndarray:
    X = df[names].to_numpy(dtype=np.float64).copy()
    if log_lengths:
        for column in LOG_LENGTH_COLUMNS:
            if column in names:
                idx = names.index(column)
                X[:, idx] = np.log1p(np.clip(X[:, idx], 0.0, None))
    return X


def make_variant_xgb(params: dict, seed: int, constraints: str | None, early_stopping: bool = False) -> XGBClassifier:
    base = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
        device=xgb_device(),
        scale_pos_weight=1.0,
        random_state=seed,
    )
    if early_stopping:
        base["early_stopping_rounds"] = 30
    if constraints:
        base["monotone_constraints"] = constraints
    base.update(params)
    return XGBClassifier(**base)


def strict_threshold(y_val: np.ndarray, p_val: np.ndarray, alpha: float = DEFAULT_ALPHA) -> dict:
    """Highest-recall threshold whose validation false-positive rate is <= alpha."""
    best = {"threshold": 0.99, "val_fpr": 0.0, "val_recall": 0.0}
    for t in np.linspace(0.01, 0.99, 197):
        pred = p_val >= t
        neg = y_val == 0
        pos = y_val == 1
        fp = int((pred & neg).sum())
        tn = int((~pred & neg).sum())
        tp = int((pred & pos).sum())
        fn = int((~pred & pos).sum())
        fpr = fp / max(fp + tn, 1)
        recall = tp / max(tp + fn, 1)
        if fpr <= alpha and recall > best["val_recall"]:
            best = {"threshold": float(t), "val_fpr": float(fpr), "val_recall": float(recall)}
    return best


def in_domain_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "mcc": float(((tp * tn - fp * fn) / np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1)))),
        "ece": float(ece(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def external_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    out = {
        "n": int(len(y)),
        "recall": float(pred[y == 1].mean()) if (y == 1).any() else None,
        "precision": float(y[pred == 1].mean()) if pred.any() else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "predicted_positive_rate": float(pred.mean()),
        "ece": float(ece(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }
    out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
    return out


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_halueval() -> pd.DataFrame:
    if not FEATURES_FULL.exists() or not CLAIM_FEATURES.exists():
        raise FileNotFoundError("Run src/features/extract_features.py and src/features/claim_features.py first")
    base = pd.read_parquet(FEATURES_FULL)
    claims = pd.read_parquet(CLAIM_FEATURES)
    df = base.merge(claims, on="sample_id", how="left", validate="one_to_one")
    if df[list(CLAIM_COLUMNS)].isna().any().any():
        raise ValueError("claim features missing after merge")
    if len(df) != 20000:
        raise ValueError(f"expected 20000 HaluEval rows, found {len(df)}")
    df[SOURCE_COLUMN] = 0.0
    return df


def load_external() -> pd.DataFrame:
    if not EXTERNAL_FEATURES.exists():
        raise FileNotFoundError("Run src/features/extract_external_features.py first")
    df = pd.read_parquet(EXTERNAL_FEATURES)
    missing = [c for c in BASE_COLUMNS + CLAIM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"external features missing columns: {missing[:5]}")
    df[SOURCE_COLUMN] = 1.0
    return df


def external_roles(df: pd.DataFrame) -> dict:
    ragtruth = df[df["source_dataset"] == "ragtruth"]
    return {
        "ragtruth_qa_test": ragtruth[(ragtruth["task"] == "qa") & (ragtruth["split"] == "test")],
        "ragtruth_all_test": ragtruth[ragtruth["split"] == "test"],
        "faithbench": df[df["source_dataset"] == "faithbench"],
    }


def m3_train_rows(df: pd.DataFrame) -> pd.DataFrame:
    rag = df[df["source_dataset"] == "ragtruth"]
    return rag[(rag["task"].isin(["summarization", "data_to_text"])) & (rag["split"] == "train")].copy()


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
def tune_variant(X, y, groups, seed: int, constraints: str | None, cfg: B6Config) -> tuple:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rs = RandomizedSearchCV(
        make_variant_xgb({}, seed, constraints), cfg.tuning_grid, n_iter=cfg.n_iter,
        cv=cv, scoring="roc_auc", n_jobs=1, random_state=seed, verbose=0,
    )
    rs.fit(X, y, groups=groups)
    logger.info(f"  seed {seed}: best {rs.best_params_} cv_auc={rs.best_score_:.4f}")
    return dict(rs.best_params_), float(rs.best_score_)


def run_b6(cfg: B6Config) -> dict:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    halueval = load_halueval()
    if "m3" in cfg.variants:
        external = load_external()
    else:
        external = pd.DataFrame(columns=[
            "sample_id", "source_dataset", "task", "split", "source_group_id", "label",
            *BASE_COLUMNS, *CLAIM_COLUMNS, SOURCE_COLUMN,
        ])
    ext = external_roles(external)
    ext_train = m3_train_rows(external)
    logger.info(
        f"HaluEval train/val/test: "
        f"{(halueval['split'] == 'train').sum()}/{(halueval['split'] == 'val').sum()}/{(halueval['split'] == 'test').sum()} | "
        f"M3 external train rows: {len(ext_train)}"
    )

    test_df = halueval[halueval["split"] == "test"].reset_index(drop=True)
    y_test = test_df["label"].to_numpy()
    val_df = halueval[halueval["split"] == "val"].reset_index(drop=True)
    y_val = val_df["label"].to_numpy()

    result_rows, external_rows, prediction_rows = [], [], []
    stats = {"mcnemar": {}, "wilcoxon": {}, "bootstrap": {}}
    strict_report = {}
    tuning_report = {}
    predictions_by_variant_seed = {}
    total_jobs = len(cfg.variants) * len(cfg.seeds)
    job = 0

    for variant in cfg.variants:
        names = variant_features(variant)
        log_lengths = variant != "m0"
        constraints = constraint_string(names) if variant != "m0" else None

        train_df = halueval[halueval["split"] == "train"].copy()
        groups = np.char.add("halu:", train_df["item_idx"].to_numpy().astype(str))
        if variant == "m3":
            extra = ext_train.copy()
            extra[SOURCE_COLUMN] = 1.0
            train_df["source_ragtruth"] = 0.0
            groups = np.concatenate([
                groups,
                np.char.add("rag:", extra["source_group_id"].astype(str)),
            ])
            train_df = pd.concat([train_df, extra], ignore_index=True)
        y_train = train_df["label"].to_numpy()
        X_train = feature_matrix(train_df, names, log_lengths)
        X_val = feature_matrix(val_df, names, log_lengths)
        X_test = feature_matrix(test_df, names, log_lengths)

        tuning_report[variant] = {}
        for seed in cfg.seeds:
            job += 1
            logger.info(
                f"[{job}/{total_jobs}] {variant} seed {seed}: tuning {cfg.n_iter} iters x 5 folds "
                f"on {len(X_train)} rows x {len(names)} features..."
            )
            t_job = time.time()
            params, cv_auc = tune_variant(X_train, y_train, groups, seed, constraints, cfg)
            tuning_report[variant][str(seed)] = {"params": params, "cv_auc": cv_auc}
            model = make_variant_xgb(params, seed, constraints, early_stopping=True)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            logger.info(f"[{job}/{total_jobs}] {variant} seed {seed}: tuned+fit in {time.time() - t_job:.0f}s")
            p_val = model.predict_proba(X_val)[:, 1]
            p_test = model.predict_proba(X_test)[:, 1]

            row = {"variant": variant, "seed": seed, "n_features": len(names), **in_domain_metrics(y_test, p_test, 0.5)}
            result_rows.append(row)
            predictions_by_variant_seed[(variant, seed)] = {"p_test": p_test, "y_test": y_test}

            for dataset_name, frame in ext.items():
                if len(frame) == 0:
                    continue
                X = feature_matrix(frame, names, log_lengths)
                y = frame["label"].to_numpy()
                p = model.predict_proba(X)[:, 1]
                external_rows.append({
                    "variant": variant, "seed": seed, "dataset": dataset_name,
                    **external_metrics(y, p, 0.5),
                })
                sample_ids = frame["sample_id"].to_numpy()
                for i in range(len(y)):
                    prediction_rows.append({
                        "variant": variant, "seed": seed, "dataset": dataset_name,
                        "sample_id": sample_ids[i], "label": int(y[i]), "score": float(p[i]),
                    })

            strict = strict_threshold(y_val, p_val, cfg.alpha)
            strict_test = in_domain_metrics(y_test, p_test, strict["threshold"])
            strict_report[f"{variant}_seed_{seed}"] = {
                **strict,
                "test_f1": strict_test["f1"], "test_recall": strict_test["recall"],
                "test_fpr": strict_test["confusion"]["fp"] / max(
                    strict_test["confusion"]["fp"] + strict_test["confusion"]["tn"], 1),
            }

            for i in range(len(y_test)):
                prediction_rows.append({
                    "variant": variant, "seed": seed, "dataset": "halueval_test",
                    "sample_id": test_df["sample_id"].iloc[i], "label": int(y_test[i]), "score": float(p_test[i]),
                })

            model_path = cfg.models_dir / f"xgboost_{variant}_seed_{seed}.joblib"
            if not cfg.smoke:
                import joblib

                joblib.dump(model, model_path)

    results_df = pd.DataFrame(result_rows)
    external_df = pd.DataFrame(external_rows)
    summary = {}
    for variant in cfg.variants:
        sub = results_df[results_df["variant"] == variant]
        entry = {"n_features": int(sub["n_features"].iloc[0])}
        for key in ("precision", "recall", "f1", "auroc", "mcc", "ece", "brier"):
            entry[f"{key}_mean"] = float(sub[key].mean())
            entry[f"{key}_std"] = float(sub[key].std())
        summary[variant] = entry

    metric_keys = ["precision", "recall", "f1", "auroc", "mcc", "ece", "brier"]
    comparison_df = pd.DataFrame(summary).T
    comparison_df.to_csv(cfg.results_dir / "b6_model_comparison.csv")
    results_df.drop(columns=["confusion"]).to_csv(cfg.results_dir / "b6_per_seed_metrics.csv", index=False)

    confusions = {}
    for row in result_rows:
        confusions[f"{row['variant']}_seed_{row['seed']}"] = row["confusion"]
    (cfg.results_dir / "b6_confusion_matrices.json").write_text(json.dumps(confusions, indent=2))

    if len(external_df) > 0:
        external_summary = external_df.groupby(["variant", "dataset"])[
            [c for c in ("n", "recall", "precision", "f1", "auroc", "predicted_positive_rate", "ece", "brier")
             if c in external_df.columns]
        ].mean()
    else:
        external_summary = pd.DataFrame()

    # Statistics: McNemar m0 vs each variant on HaluEval test (first seed), bootstrap, Wilcoxon.
    base_variant = cfg.variants[0]
    seed0 = cfg.seeds[0]
    if (base_variant, seed0) in predictions_by_variant_seed:
        p0 = predictions_by_variant_seed[(base_variant, seed0)]["p_test"]
        pred0 = (p0 >= 0.5).astype(int)
        for variant in cfg.variants[1:]:
            key = (variant, seed0)
            if key not in predictions_by_variant_seed:
                continue
            pv = predictions_by_variant_seed[key]["p_test"]
            predv = (pv >= 0.5).astype(int)
            b = int(((pred0 == 0) & (predv == 1)).sum())
            c = int(((pred0 == 1) & (predv == 0)).sum())
            stats["mcnemar"][f"{base_variant}_vs_{variant}"] = float(
                mcnemar([[0, b], [c, 0]], exact=False, correction=True).pvalue
            )
            stats["bootstrap"][f"{variant}_seed_{seed0}"] = bootstrap_ci(
                predictions_by_variant_seed[key]["y_test"], predv, pv
            )
        for variant in cfg.variants[1:]:
            f1_base = [r["f1"] for r in result_rows if r["variant"] == base_variant]
            f1_var = [r["f1"] for r in result_rows if r["variant"] == variant]
            if len(f1_base) == len(f1_var) == len(cfg.seeds) and len(set(f1_base)) > 1:
                try:
                    stats["wilcoxon"][f"{base_variant}_vs_{variant}"] = float(
                        scipy_stats.wilcoxon(f1_base, f1_var).pvalue
                    )
                except ValueError:
                    stats["wilcoxon"][f"{base_variant}_vs_{variant}"] = None

    if len(external_summary) > 0:
        external_summary.to_csv(cfg.results_dir / "b6_external_metrics.csv")
    (cfg.results_dir / "b6_statistical_tests.json").write_text(json.dumps(stats, indent=2))
    (cfg.results_dir / "b6_strict_mode.json").write_text(
        json.dumps({"alpha": cfg.alpha, "per_run": strict_report}, indent=2)
    )
    (cfg.results_dir / "b6_tuning.json").write_text(json.dumps(tuning_report, indent=2))

    feature_names_path = cfg.results_dir / "b6_feature_names.json"
    feature_names_path.write_text(json.dumps({v: variant_features(v) for v in cfg.variants}, indent=2))

    if prediction_rows:
        pd.DataFrame(prediction_rows).to_parquet(cfg.results_dir / "b6_predictions.parquet", index=False)

    config_out = {
        "schema": "b6-config-v1",
        "model_name": MODEL_NAME,
        "model_full_name": MODEL_FULL_NAME,
        "component_map": {
            "m0": "standard tuned XGBoost (reference)",
            "m1": "log-scaled length features + domain monotone constraints",
            "m2": "m1 + claim-level NLI aggregation features",
            "m3": MODEL_NAME + " (m2 + RAGTruth non-QA multi-source training)",
        },
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "variants": list(cfg.variants),
        "seeds": cfg.seeds,
        "n_iter_tuning": cfg.n_iter,
        "alpha_strict": cfg.alpha,
        "device": xgb_device(),
        "base_features": BASE_COLUMNS,
        "claim_features": list(CLAIM_COLUMNS),
        "source_indicator": SOURCE_COLUMN,
        "log_length_columns": list(LOG_LENGTH_COLUMNS),
        "monotone_constraints": CONSTRAINTS,
    }
    (cfg.results_dir / "b6_run_config.json").write_text(json.dumps(config_out, indent=2))

    logger.info(f"Saved {MODEL_NAME} ({MODEL_FULL_NAME}) artifacts to {cfg.results_dir}")
    print(f"\n{MODEL_FULL_NAME} ({MODEL_NAME}) — test-set comparison")
    print(comparison_df.round(4).to_string())
    return {"comparison": summary, "external": external_summary.to_dict(), "stats": stats}


# --------------------------------------------------------------------------
# Smoke test (synthetic; no downloads)
# --------------------------------------------------------------------------
def build_synthetic(n_groups: int = 40, seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_groups):
        split = "train" if i < int(n_groups * 0.6) else ("val" if i < int(n_groups * 0.8) else "test")
        for label, suffix in ((0, "c"), (1, "h")):
            base = {c: float(rng.random()) for c in BASE_COLUMNS}
            claim = {c: float(rng.random()) for c in CLAIM_COLUMNS}
            base["overlap_answer_context"] = float(rng.random()) * (0.05 if label else 0.9)
            base["nli_ctx_entails_ans"] = 0.6 + 0.3 * (1 - label) * float(rng.random())
            base["nli_ctx_contradicts_ans"] = 0.1 + 0.6 * label * float(rng.random())
            claim["claim_max_contradicts"] = 0.1 + 0.7 * label * float(rng.random())
            claim["claim_min_entails"] = 0.8 * (1 - label) * float(rng.random())
            rows.append({"sample_id": f"q_{i}_{suffix}", "item_idx": i, "label": label, "split": split, **base, **claim})
    halueval = pd.DataFrame(rows)
    halueval[SOURCE_COLUMN] = 0.0
    ext_rows = []
    for j in range(80):
        base = {c: float(rng.random()) for c in BASE_COLUMNS}
        claim = {c: float(rng.random()) for c in CLAIM_COLUMNS}
        ext_rows.append({
            "sample_id": f"ext_{j}", "source_dataset": "ragtruth", "task": "qa" if j % 2 else "summarization",
            "split": "train" if j < 40 else "test", "official_split": "train" if j < 40 else "test",
            "source_group_id": f"g{j}", "item_idx": j, "label": int(rng.random() > 0.5), **base, **claim,
        })
    external = pd.DataFrame(ext_rows)
    external[SOURCE_COLUMN] = 1.0
    return halueval, external


def run_smoke() -> dict:
    halueval, external = build_synthetic()
    cfg = B6Config(
        results_dir=RESULTS_DIR / "b6_smoke",
        models_dir=MODELS_DIR / "b6_smoke",
        variants=("m0", "m2", "m3"),
        seeds=[42],
        n_iter=2,
        tuning_grid={"max_depth": [3, 4], "learning_rate": [0.1], "n_estimators": [40], "subsample": [0.9], "colsample_bytree": [0.9]},
        smoke=True,
    )
    tmp = RESULTS_DIR / "b6_smoke"
    tmp.mkdir(parents=True, exist_ok=True)
    halueval_path = tmp / "halueval_synth.parquet"
    external_path = tmp / "external_synth.parquet"
    halueval.to_parquet(halueval_path, index=False)
    external.to_parquet(external_path, index=False)

    # Synthetic frames carry the base and claim columns directly, so bypass the
    # real loaders (which merge separate feature files).
    namespace = globals()
    originals = {key: namespace[key] for key in ("FEATURES_FULL", "EXTERNAL_FEATURES", "load_halueval", "load_external")}
    try:
        namespace["FEATURES_FULL"] = halueval_path
        namespace["EXTERNAL_FEATURES"] = external_path
        namespace["load_halueval"] = lambda: pd.read_parquet(halueval_path)
        namespace["load_external"] = lambda: pd.read_parquet(external_path)
        return run_b6(cfg)
    finally:
        namespace.update(originals)


def main() -> None:
    parser = argparse.ArgumentParser(description="B6: modified XGBoost variants")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke()
        print("B6 smoke test complete.")
        return

    cfg = B6Config(
        variants=tuple(v.strip() for v in args.variants.split(",") if v.strip()),
        seeds=[int(s) for s in args.seeds.split(",") if s.strip()],
        n_iter=args.n_iter,
        alpha=args.alpha,
    )
    if args.results_dir:
        cfg.results_dir = args.results_dir
    if args.models_dir:
        cfg.models_dir = args.models_dir
    unknown = [v for v in cfg.variants if v not in DEFAULT_VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    run_b6(cfg)


if __name__ == "__main__":
    main()
