"""
HaluRISC artifact manifest generator (blueprint A18 / roadmap B6).

Writes artifacts/results/manifest.json with dataset hashes, split report,
package versions, hardware/software info, model/feature versions, and the
list of produced artifacts. Colab-safe: repo-root-relative paths only.

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
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.config import ARTIFACTS_DIR, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("make_manifest")

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


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main():
    params = read_json(MODELS_DIR / "params.json") or {}
    split_report = read_json(ARTIFACTS_DIR / "split_integrity_report.json")
    nli_used = read_json(DATA_PROCESSED / "nli_model_used.json")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_version": params.get("model_version"),
        "feature_version": "course-v1.0",
        "n_features": params.get("n_features"),
        "nli_model": params.get("nli_model"),
        "nli_provenance": nli_used,
        "split_report": split_report,
        "dataset_sha256": {
            "qa_clean.parquet": sha256(SHA_FILES[0]),
            "features_full.parquet": sha256(SHA_FILES[1]),
            "model_xgboost_raw.joblib": sha256(SHA_FILES[2]),
            "model_xgboost_calibrated.joblib": sha256(SHA_FILES[3]),
            "split_indices.json": sha256(SHA_FILES[4]),
            "split_integrity_report.json": sha256(SHA_FILES[5]),
        },
        "versions": versions(),
        "hardware": hardware(),
        "artifacts": sorted(
            str(p.relative_to(ARTIFACTS_DIR.parent))
            for p in [
                *ARTIFACTS_DIR.rglob("*"),
                *DATA_PROCESSED.glob("*.parquet"),
                DATA_PROCESSED / "nli_model_used.json",
            ]
            if p.is_file()
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Saved manifest to {out}")
    return manifest


if __name__ == "__main__":
    main()
