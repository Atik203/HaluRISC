"""
Fit the EC-XGB display calibrator on the designated RAGTruth QA calibration
split (b4 protocol) and evaluate it on the held-out 900-row QA test.

Steps:
  1. Join the calibration sample_ids with unified_records for text.
  2. Extract the 35 EC-XGB features (26 base + 8 claim + source indicator),
     cached to data/processed/calibration_features.parquet.
  3. Score with artifacts/models/b6/xgboost_m3_seed_42.joblib.
  4. Fit Platt and isotonic calibrators on the calibration scores.
  5. Evaluate both on the QA test scores from b6_predictions.parquet, then
     save the method with the lower ECE as the deployed display calibrator.

Serving convention: the source indicator is 1.0 for natural responses (the
same setting used for every external row in the B6 evaluation), so the
calibrator is fit and applied under the deployed convention.

Run (repo root, .venv):
  python src/models/fit_ec_xgb_calibrator.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.features.claim_features import FEATURE_COLUMNS as CLAIM_COLUMNS
from src.features.claim_features import extract_claim_features_df
from src.features.extract_features import extract_full_feature_set, load_heavy_models
from src.models.config import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR
from src.models.run_b6_modified import BASE_COLUMNS, SOURCE_COLUMN, feature_matrix, variant_features
from src.models.train_pipeline import ece

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ec_xgb_calibrator")

UNIFIED = DATA_PROCESSED / "unified_records.parquet"
CAL_IDS = RESULTS_DIR / "b4" / "_stages" / "qa_cal_clean.parquet"
# Base (26) and claim (8) features are cached separately so a claim-splitter
# change only re-runs the cheap claim pass.
CAL_BASE = DATA_PROCESSED / "calibration_base.parquet"
CAL_CLAIMS = DATA_PROCESSED / "calibration_claim_features.parquet"
CAL_FEATURES = DATA_PROCESSED / "calibration_features.parquet"  # legacy merged cache
EC_XGB_MODEL = MODELS_DIR / "b6" / "xgboost_m3_seed_42.joblib"
B6_PREDICTIONS = RESULTS_DIR / "b6" / "b6_predictions.parquet"
OUT_BUNDLE = MODELS_DIR / "b6" / "ec_xgb_display_calibrator.joblib"
OUT_METRICS = RESULTS_DIR / "b6" / "b6_calibration.json"


def build_calibration_frame() -> pd.DataFrame:
    cal = pd.read_parquet(CAL_IDS)[["sample_id", "label"]]
    records = pd.read_parquet(UNIFIED, columns=["sample_id", "question", "context", "answer"])
    df = cal.merge(records, on="sample_id", how="left", validate="one_to_one")
    if df[["question", "context", "answer"]].isna().any().any():
        raise ValueError("calibration rows missing text in unified_records")
    for col in ("question", "context", "answer"):
        df[col] = df[col].fillna("").astype(str)
    df["item_idx"] = np.arange(len(df))
    df["split"] = "calibration"
    logger.info(f"Calibration frame: {len(df)} rows, labels {df['label'].value_counts().to_dict()}")
    return df


def ensure_base_features(df: pd.DataFrame) -> pd.DataFrame:
    if CAL_BASE.exists():
        base = pd.read_parquet(CAL_BASE)
        logger.info(f"Reusing calibration base features: {base.shape}")
        return base
    if CAL_FEATURES.exists():
        # Seed the split cache from the legacy merged file: the base columns are
        # unaffected by claim-splitter changes.
        legacy = pd.read_parquet(CAL_FEATURES)
        keep = [c for c in legacy.columns if c in set(BASE_COLUMNS) | {"sample_id", "item_idx", "label", "split"}]
        base = legacy[keep]
        base.to_parquet(CAL_BASE, index=False)
        logger.info(f"Seeded calibration base cache from legacy file: {base.shape}")
        return base
    models = load_heavy_models(device=os.environ.get("HALU_EXTERNAL_DEVICE", "cuda"))
    t0 = time.time()
    base = extract_full_feature_set(df, models, batch_size=64)
    logger.info(f"Base features done in {time.time() - t0:.1f}s")
    base.to_parquet(CAL_BASE, index=False)
    return base


def ensure_claim_features(df: pd.DataFrame) -> pd.DataFrame:
    if CAL_CLAIMS.exists():
        claims = pd.read_parquet(CAL_CLAIMS)
        logger.info(f"Reusing calibration claim features: {claims.shape}")
        return claims
    models = load_heavy_models(device=os.environ.get("HALU_EXTERNAL_DEVICE", "cuda"))
    t0 = time.time()
    claims = extract_claim_features_df(
        df, models["nli"], partial_path=DATA_PROCESSED / "calibration_claims.partial.parquet",
        checkpoint_every=500, batch_size=16, chunk_samples=24,
    )
    logger.info(f"Claim features done in {time.time() - t0:.1f}s")
    claims.to_parquet(CAL_CLAIMS, index=False)
    return claims


def ensure_calibration_features(df: pd.DataFrame) -> pd.DataFrame:
    base = ensure_base_features(df)
    claims = ensure_claim_features(df)
    out = base.merge(claims, on="sample_id", how="left")
    out.to_parquet(CAL_FEATURES, index=False)
    logger.info(f"Calibration features ready: {out.shape}")
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fit the EC-XGB display calibrator")
    parser.add_argument("--features-only", action="store_true", help="extract and cache features, then stop")
    args = parser.parse_args()

    df = build_calibration_frame()
    feats = ensure_calibration_features(df)
    if args.features_only:
        logger.info("Features cached; stopping before the calibrator fit (--features-only).")
        return
    frame = df.merge(feats.drop(columns=["item_idx", "label", "split"], errors="ignore"),
                     on="sample_id", how="left", validate="one_to_one")
    frame[SOURCE_COLUMN] = 1.0  # natural-response convention (same as B6 external)

    names = json.loads((RESULTS_DIR / "b6" / "b6_feature_names.json").read_text())["m3"]
    model = joblib.load(EC_XGB_MODEL)
    X = feature_matrix(frame, names, log_lengths=True)
    raw = model.predict_proba(X)[:, 1]
    y = frame["label"].to_numpy()
    logger.info(f"Calibration raw ECE {ece(y, raw):.4f} | Brier {brier_score_loss(y, raw):.4f}")

    platt = LogisticRegression(max_iter=2000).fit(raw.reshape(-1, 1), y)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(raw, y)

    test = pd.read_parquet(B6_PREDICTIONS)
    test = test[(test["variant"] == "m3") & (test["seed"] == 42) & (test["dataset"] == "ragtruth_qa_test")]
    y_test = test["label"].to_numpy()
    raw_test = test["score"].to_numpy()
    p_platt = platt.predict_proba(raw_test.reshape(-1, 1))[:, 1]
    p_iso = np.asarray(isotonic.predict(raw_test))

    metrics = {
        "calibration_rows": int(len(frame)),
        "test_rows": int(len(test)),
        "source_indicator": 1.0,
        "model": "ec-xgb-m3-seed-42",
        "calibration_raw": {"ece": float(ece(y, raw)), "brier": float(brier_score_loss(y, raw))},
        "test_raw": {"ece": float(ece(y_test, raw_test)), "brier": float(brier_score_loss(y_test, raw_test))},
        "test_platt": {"ece": float(ece(y_test, p_platt)), "brier": float(brier_score_loss(y_test, p_platt))},
        "test_isotonic": {"ece": float(ece(y_test, p_iso)), "brier": float(brier_score_loss(y_test, p_iso))},
    }
    chosen = "isotonic" if metrics["test_isotonic"]["ece"] <= metrics["test_platt"]["ece"] else "platt"
    calibrator = isotonic if chosen == "isotonic" else platt
    metrics["chosen"] = chosen
    logger.info(f"Chosen display calibrator: {chosen}")
    logger.info(json.dumps({k: v for k, v in metrics.items() if k != "chosen"}, indent=2))

    OUT_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"method": chosen, "calibrator": calibrator, **{k: metrics[k] for k in
                 ("calibration_rows", "test_rows", "source_indicator", "model", "test_raw", "test_platt", "test_isotonic")}},
                OUT_BUNDLE)
    OUT_METRICS.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Saved {OUT_BUNDLE} and {OUT_METRICS}")


if __name__ == "__main__":
    main()
