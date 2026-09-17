"""
HaluRISC artifact manifest generator (blueprint §11 / roadmap B6).

Writes artifacts/results/manifest.json with:
  - dataset hashes (processed + raw sources with revision files)
  - split report (split hash via split_indices.json)
  - seeds [42, 123, 456], feature groups, model/feature versions
  - package versions + hardware (CPU/RAM/GPU)
  - git commit (when run inside a clone) OR source_fingerprint (sha256 over
    the notebook cell-3 embedded HASHES, passed via HALU_SOURCE_FINGERPRINT;
    this fingerprints the exact shipped source even on Colab without git)
  - HALU_* environment configuration (non-secret; API keys are never included)
  - the list of produced artifacts (internal checkpoints excluded)

Colab-safe: repo-root-relative paths only.

Run (repo root, .venv or Colab):
  python src/models/make_manifest.py
"""

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.config import (  # noqa: E402
    ARTIFACTS_DIR,
    DATA_PROCESSED,
    MODELS_DIR,
    RESULTS_DIR,
    SEEDS,
)
from src.models.train_pipeline import FEATURE_GROUPS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("make_manifest")

RAW_DIR = ARTIFACTS_DIR.parent / "data" / "raw"

# Internal/non-publication paths never listed in the manifest (or future zips).
EXCLUDE_FRAGMENTS = ("_stages", "b2_smoke_test", "b2_test_tmp", "b5_crash.log", "__pycache__")

SHA_FILES = [
    DATA_PROCESSED / "qa_clean.parquet",
    DATA_PROCESSED / "features_full.parquet",
    MODELS_DIR / "model_xgboost_raw.joblib",
    MODELS_DIR / "model_xgboost_calibrated.joblib",
    ARTIFACTS_DIR / "split_indices.json",
    ARTIFACTS_DIR / "split_integrity_report.json",
]


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def versions() -> dict:
    def ver(name: str):
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": ver("torch"),
        "numpy": ver("numpy"),
        "pandas": ver("pandas"),
        "scikit_learn": ver("sklearn"),
        "xgboost": ver("xgboost"),
        "shap": ver("shap"),
        "spacy": ver("spacy"),
        "sentence_transformers": ver("sentence_transformers"),
        "fastapi": ver("fastapi"),
        "pyyaml": ver("yaml"),
    }


def hardware() -> dict:
    hw = {"cpu": platform.processor(), "cuda": False, "gpu_name": None, "gpu_total_memory_gb": None}
    try:
        import torch

        if torch.cuda.is_available():
            hw["cuda"] = True
            hw["gpu_name"] = torch.cuda.get_device_name(0)
            hw["gpu_total_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    except Exception:
        pass
    try:
        import psutil

        hw["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 2)
    except Exception:
        hw["ram_total_gb"] = None
    return hw


def raw_hashes() -> dict:
    """sha256 of every raw input file (incl. download revision.json files)."""
    if not RAW_DIR.exists():
        return {}
    out = {}
    for p in sorted(RAW_DIR.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(RAW_DIR.parent))] = sha256(p)
    return out


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main():
    params = read_json(MODELS_DIR / "params.json") or {}
    split_report = read_json(ARTIFACTS_DIR / "split_integrity_report.json")
    nli_used = read_json(DATA_PROCESSED / "nli_model_used.json")

    def sha_or_none(rel_path: Path):
        return sha256(rel_path) if rel_path.exists() else None

    b_paths = {
        "b2_run_config.json": RESULTS_DIR / "b2" / "b2_run_config.json",
        "b2_model_comparison.json": RESULTS_DIR / "b2" / "b2_model_comparison.json",
        "b2_predictions.parquet": RESULTS_DIR / "b2" / "b2_predictions.parquet",
        "b2_xgboost_seed_42.joblib": MODELS_DIR / "b2" / "xgboost_seed_42.joblib",
        "b3_predictions.parquet": RESULTS_DIR / "b3" / "b3_predictions.parquet",
        "b3_dataset_metrics.json": RESULTS_DIR / "b3" / "b3_dataset_metrics.json",
        "b3_bootstrap_cis.json": RESULTS_DIR / "b3" / "b3_bootstrap_cis.json",
        "b3_run_config.json": RESULTS_DIR / "b3" / "b3_run_config.json",
        "b4_predictions.parquet": RESULTS_DIR / "b4" / "b4_predictions.parquet",
        "b4_calibration_metrics.json": RESULTS_DIR / "b4" / "b4_calibration_metrics.json",
        "b4_target_calibration.json": RESULTS_DIR / "b4" / "b4_target_calibration.json",
        "b4_calibrator_platt_source_seed_42.joblib": MODELS_DIR / "b4" / "calibrator_platt_source_seed_42.joblib",
        "b4_calibrator_isotonic_source_seed_42.joblib": MODELS_DIR / "b4" / "calibrator_isotonic_source_seed_42.joblib",
        "unified_records.parquet": DATA_PROCESSED / "unified_records.parquet",
        "b3_external_features.parquet": DATA_PROCESSED / "b3_external_features.parquet",
        "b5_run_config.json": RESULTS_DIR / "b5" / "b5_run_config.json",
        "b5_feature_importance.json": RESULTS_DIR / "b5" / "b5_feature_importance.json",
        "b5_neutralization.json": RESULTS_DIR / "b5" / "b5_neutralization.json",
        "b5_stability_bootstrap.json": RESULTS_DIR / "b5" / "b5_stability_bootstrap.json",
        "b5_perturbation_aggregates.csv": RESULTS_DIR / "b5" / "b5_perturbation_aggregates.csv",
        "b5_review_cases.csv": RESULTS_DIR / "b5" / "b5_review_cases.csv",
        "b5_failure_cases.json": RESULTS_DIR / "b5" / "b5_failure_cases.json",
    }

    env_config = {
        name: os.environ[name]
        for name in ("HALU_API_DEVICE", "HALU_API_PRELOAD", "HALU_XGB_DEVICE",
                     "HALU_NLI_MODEL", "HALU_JUDGE_N")
        if name in os.environ
    }
    source_fingerprint = os.environ.get("HALU_SOURCE_FINGERPRINT")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "source_fingerprint": source_fingerprint,
        "model_version": params.get("model_version"),
        "feature_version": "course-v1.0",
        "n_features": params.get("n_features"),
        "nli_model": params.get("nli_model"),
        "nli_provenance": nli_used,
        "seeds": list(SEEDS),
        "feature_groups": FEATURE_GROUPS,
        "split_report": split_report,
        "dataset_sha256": {
            "qa_clean.parquet": sha256(SHA_FILES[0]),
            "features_full.parquet": sha256(SHA_FILES[1]),
            "model_xgboost_raw.joblib": sha256(SHA_FILES[2]),
            "model_xgboost_calibrated.joblib": sha256(SHA_FILES[3]),
            "split_indices.json": sha256(SHA_FILES[4]),
            "split_integrity_report.json": sha256(SHA_FILES[5]),
        },
        "raw_sha256": raw_hashes(),
        "b_artifacts_sha256": {name: sha_or_none(p) for name, p in b_paths.items()},
        "versions": versions(),
        "hardware": hardware(),
        "env": env_config,
        "artifacts": sorted(
            str(p.relative_to(ARTIFACTS_DIR.parent))
            for p in [
                *ARTIFACTS_DIR.rglob("*"),
                *DATA_PROCESSED.glob("*.parquet"),
                DATA_PROCESSED / "nli_model_used.json",
            ]
            if p.is_file() and not any(frag in str(p) for frag in EXCLUDE_FRAGMENTS)
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Saved manifest to {out} (git_commit={manifest['git_commit']}, "
                f"source_fingerprint={'set' if source_fingerprint else None}, "
                f"raw files={len(manifest['raw_sha256'])})")
    return manifest


if __name__ == "__main__":
    main()
