"""
Add the EC-XGB (evidence-consistent XGBoost) evidence to the frozen manifest.

The original B-run freeze stays untouched. This script appends:

  ec_xgb_artifacts_sha256  SHA-256 for every file under artifacts/results/b6/
                           and artifacts/models/b6/
  ec_xgb                   model identity, components, and the headline
                           numbers that the manuscript reports, read from the
                           b6 result files

Run (repo root, .venv):
  python src/models/freeze_ec_xgb_manifest.py
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.models.config import MODELS_DIR, RESULTS_DIR, ROOT

FROZEN = ROOT / "docs" / "manifest.frozen.json"
B6_RESULTS = RESULTS_DIR / "b6"
B6_MODELS = MODELS_DIR / "b6"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def main() -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))

    hashes = {}
    for directory in (B6_RESULTS, B6_MODELS):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)
    frozen["ec_xgb_artifacts_sha256"] = hashes

    comparison = pd.read_csv(B6_RESULTS / "b6_model_comparison.csv", index_col=0)
    external = pd.read_csv(B6_RESULTS / "b6_external_metrics.csv")
    stats = json.loads((B6_RESULTS / "b6_statistical_tests.json").read_text())
    strict = json.loads((B6_RESULTS / "b6_strict_mode.json").read_text())
    calibration = json.loads((B6_RESULTS / "b6_calibration.json").read_text())

    def row(variant: str, column: str) -> float:
        return float(comparison.loc[variant, column])

    def ext(dataset: str, variant: str, column: str) -> float:
        match = external[(external["dataset"] == dataset) & (external["variant"] == variant)]
        return float(match[column].iloc[0])

    strict_runs = list(strict["per_run"].values())
    frozen["ec_xgb"] = {
        "model_name": "EC-XGB",
        "model_full_name": "Evidence-Consistent XGBoost",
        "api_version": "b6-ec-xgb-v1.0",
        "components": {
            "m1": "log-scaled length features + monotone evidence constraints",
            "m2": "m1 + 8 claim-level NLI aggregates (top-4 sentences, max 6 claims, 400-char cap)",
            "m3": "m2 + RAGTruth non-QA training rows with a source indicator (35 features)",
        },
        "in_domain": {
            "m0_f1": row("m0", "f1_mean"),
            "m0_auroc": row("m0", "auroc_mean"),
            "m3_f1": row("m3", "f1_mean"),
            "m3_auroc": row("m3", "auroc_mean"),
            "m3_mcc": row("m3", "mcc_mean"),
            "m3_ece": row("m3", "ece_mean"),
            "mcnemar_m0_vs_m3_p": stats["mcnemar"]["m0_vs_m3"],
            "wilcoxon_m0_vs_m3_p": stats["wilcoxon"]["m0_vs_m3"],
        },
        "shift": {
            "ragtruth_all": {
                "standard_flag_rate": ext("ragtruth_all_test", "m0", "predicted_positive_rate"),
                "standard_auroc": ext("ragtruth_all_test", "m0", "auroc"),
                "standard_ece": ext("ragtruth_all_test", "m0", "ece"),
                "ec_xgb_flag_rate": ext("ragtruth_all_test", "m3", "predicted_positive_rate"),
                "ec_xgb_auroc": ext("ragtruth_all_test", "m3", "auroc"),
                "ec_xgb_ece": ext("ragtruth_all_test", "m3", "ece"),
            },
            "ragtruth_qa": {
                "standard_f1": ext("ragtruth_qa_test", "m0", "f1"),
                "ec_xgb_f1": ext("ragtruth_qa_test", "m3", "f1"),
                "standard_ece": ext("ragtruth_qa_test", "m0", "ece"),
                "ec_xgb_ece": ext("ragtruth_qa_test", "m3", "ece"),
            },
            "faithbench": {
                "standard_flag_rate": ext("faithbench", "m0", "predicted_positive_rate"),
                "standard_f1": ext("faithbench", "m0", "f1"),
                "standard_auroc": ext("faithbench", "m0", "auroc"),
                "ec_xgb_flag_rate": ext("faithbench", "m3", "predicted_positive_rate"),
                "ec_xgb_f1": ext("faithbench", "m3", "f1"),
                "ec_xgb_auroc": ext("faithbench", "m3", "auroc"),
            },
        },
        "strict_mode": {
            "alpha": strict["alpha"],
            "test_fpr_range": [min(r["test_fpr"] for r in strict_runs), max(r["test_fpr"] for r in strict_runs)],
            "test_recall_range": [min(r["test_recall"] for r in strict_runs), max(r["test_recall"] for r in strict_runs)],
        },
        "display_calibration": calibration,
        "updated_at_git_commit": git_commit(),
    }

    FROZEN.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"Updated {FROZEN} with {len(hashes)} EC-XGB artifact hashes")
    print(json.dumps(frozen["ec_xgb"]["in_domain"], indent=1))


if __name__ == "__main__":
    main()
