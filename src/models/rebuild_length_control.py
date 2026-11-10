"""WP7 supporting check: length control on the fixed-hp m3 predictions.

Reuses the exact binning, 247-rows-per-bin subset, length-only recipe, and
exact-length matching from src/models/review_length_control.py, but reads the
rebuilt fixed-hp predictions instead of the tuned B6 run.

Writes (under artifacts/results/rebuild/):
  length_control_fixed_hp.json
  length_control_fixed_hp_summary.csv
  length_control_fixed_hp_bins.csv
  length_control_fixed_hp_subset_ids.csv
  length_control_fixed_hp_matched_ids.csv

Run (repo root):
  python src/models/rebuild_length_control.py
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.models.review_length_control as lc  # noqa: E402

FINAL = ROOT / "artifacts" / "results" / "rebuild"
STAGING = FINAL / "_length_control_tmp"


def main() -> None:
    lc.B6 = FINAL / "fixed_hp" / "fixed_hp_predictions.parquet"
    lc.OUT = STAGING
    lc.main()

    bundle_path = STAGING / "length_control.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["meta"]["source"] = "src/models/rebuild_length_control.py"
    bundle["meta"]["ec_xgb"] = "fixed_hp m3, seeds 42/123/456"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    mapping = {
        "length_control.json": "length_control_fixed_hp.json",
        "length_control_summary.csv": "length_control_fixed_hp_summary.csv",
        "length_control_bins.csv": "length_control_fixed_hp_bins.csv",
        "length_control_subset_ids.csv": "length_control_fixed_hp_subset_ids.csv",
        "length_control_matched_ids.csv": "length_control_fixed_hp_matched_ids.csv",
    }
    for source, target in mapping.items():
        (STAGING / source).replace(FINAL / target)

    shutil.rmtree(STAGING, ignore_errors=True)
    print(f"Wrote {FINAL / 'length_control_fixed_hp.json'}")


if __name__ == "__main__":
    main()
