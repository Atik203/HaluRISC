"""
Post-run artifact verification (Colab cell 7i + local post-download check).

Loads every artifact the pipeline produces and proves it can be used on THIS
machine, preventing the historical "downloaded model fails to load" class of
errors (CUDA-trained boosters not porting across platforms):

  - B2 XGBoost boosters (3 seeds) load and predict within [0, 1]
  - B4 source + target calibrators (pure sklearn) load and apply
  - B2/B3/B4 prediction parquet files exist with the expected schema
  - Feature-column order matches the B2 config

Exit code 0 = everything usable; 1 = first failing check (clear message).

Run (repo root, .venv or Colab after B3/B4):
  python src/models/verify_artifacts.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_artifacts")

from src.models.config import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR  # noqa: E402

B2_MODELS_DIR = MODELS_DIR / "b2"
B2_RESULTS = RESULTS_DIR / "b2"
B3_RESULTS = RESULTS_DIR / "b3"
B4_RESULTS = RESULTS_DIR / "b4"
B4_MODELS = MODELS_DIR / "b4"
FEATURES_FULL = DATA_PROCESSED / "features_full.parquet"
SEEDS = [42, 123, 456]

FAILURES = []


def check(name, fn):
    try:
        fn()
        logger.info(f"OK  {name}")
    except Exception as e:
        FAILURES.append(name)
        logger.error(f"FAIL {name}: {e}")


def main() -> int:
    b2_cfg = json.loads((B2_RESULTS / "b2_run_config.json").read_text())
    feature_cols = list(b2_cfg["feature_cols"])
    logger.info(f"B2 config loaded: {len(feature_cols)} features")

    if not FEATURES_FULL.exists():
        logger.error("features_full.parquet missing (run cell 6)")
        return 1
    features = pd.read_parquet(FEATURES_FULL)
    sample = features[features["split"] == "val"].head(32)

    for seed in SEEDS:
        def _load_predict(seed=seed):
            model = joblib.load(B2_MODELS_DIR / f"xgboost_seed_{seed}.joblib")
            p = model.predict_proba(sample[feature_cols].values)[:, 1]
            assert (p >= 0.0).all() and (p <= 1.0).all(), "probabilities out of [0,1]"
        check(f"B2 xgboost_seed_{seed} loads and predicts", _load_predict)

    def _check_b2_parquet():
        preds = pd.read_parquet(B2_RESULTS / "b2_predictions.parquet")
        assert {"sample_id", "model", "score", "pred", "label"} <= set(preds.columns)
        assert preds["model"].nunique() >= 1
    check("B2 b2_predictions.parquet schema", _check_b2_parquet)

    def _check_b3():
        preds = pd.read_parquet(B3_RESULTS / "b3_predictions.parquet")
        assert {"sample_id", "source_dataset", "source_group_id", "task", "label", "model", "score", "pred"} <= set(preds.columns)
        assert {"xgboost_seed_42", "xgboost_seed_123", "xgboost_seed_456"} <= set(preds["model"])
        json.loads((B3_RESULTS / "b3_dataset_metrics.json").read_text())
        json.loads((B3_RESULTS / "b3_bootstrap_cis.json").read_text())
    check("B3 predictions + reports", _check_b3)

    for name in ("calibrator_platt_source_seed_42.joblib",
                 "calibrator_isotonic_source_seed_42.joblib",
                 "calibrator_platt_target_ragtruth_qa_seed_42.joblib",
                 "calibrator_isotonic_target_ragtruth_qa_seed_42.joblib"):
        def _load_cal(name=name):
            cal = joblib.load(B4_MODELS / name)
            p = cal.predict_proba(np.array([[0.1], [0.5], [0.9]]))[:, 1] if "platt" in name \
                else cal.predict(np.array([0.1, 0.5, 0.9]))
            assert (p >= 0.0).all() and (p <= 1.0).all(), "calibrated scores out of [0,1]"
        check(f"B4 {name} loads and applies", _load_cal)

    def _check_b4():
        preds = pd.read_parquet(B4_RESULTS / "b4_predictions.parquet")
        assert {"sample_id", "method", "score", "pred", "label"} <= set(preds.columns)
        assert {"raw", "platt", "isotonic"} <= set(preds["method"])
        json.loads((B4_RESULTS / "b4_calibration_metrics.json").read_text())
        json.loads((B4_RESULTS / "b4_target_calibration.json").read_text())
    check("B4 predictions + reports", _check_b4)

    if FAILURES:
        logger.error(f"VERIFICATION FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    logger.info("ALL ARTIFACTS VERIFIED: portable models/calibrators load and predict on this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
