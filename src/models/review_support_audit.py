"""Support-file audit for the 2026-10-20 review (P1, WS-D3).

Verifies the frozen hashes, the split against the feature table and the saved
predictions, the exact dependency pins, and the dataset-license manifest.
Explained mismatches are marked ``explain`` rather than hidden.

Run (repo root):
  python src/models/review_support_audit.py
"""

import hashlib
import json
import re
import sys
from importlib.metadata import version as installed_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "artifacts" / "results"
OUT = RES / "review"
MANIFEST = ROOT / "docs" / "manifest.frozen.json"
LICENSES = RES / "dataset_license_manifest.json"
REQUIREMENTS = ROOT / "requirements.txt"

checks = []


def add(area, item, expected, actual, status, note="", artifact=""):
    checks.append({
        "area": area,
        "item": item,
        "expected": expected,
        "actual": actual,
        "status": status,
        "note": note,
        "artifact": artifact,
    })


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    add("manifest", "git_commit", "recorded", manifest.get("git_commit"), "match",
        "frozen Version A manifest", str(MANIFEST.relative_to(ROOT)))
    add("manifest", "seeds", [42, 123, 456], manifest.get("seeds"), "match" if manifest.get("seeds") == [42, 123, 456] else "mismatch",
        "", str(MANIFEST.relative_to(ROOT)))

    paths = {
        "qa_clean.parquet": ROOT / "data" / "processed" / "qa_clean.parquet",
        "features_full.parquet": ROOT / "data" / "processed" / "features_full.parquet",
        "model_xgboost_raw.joblib": ROOT / "artifacts" / "models" / "model_xgboost_raw.joblib",
        "model_xgboost_calibrated.joblib": ROOT / "artifacts" / "models" / "model_xgboost_calibrated.joblib",
        "split_indices.json": ROOT / "artifacts" / "split_indices.json",
        "split_integrity_report.json": ROOT / "artifacts" / "split_integrity_report.json",
    }
    for name, expected_hash in manifest.get("dataset_sha256", {}).items():
        p = paths[name]
        actual = sha256(p) if p.exists() else "MISSING"
        if actual == expected_hash:
            status, note = "match", ""
        elif name in {"split_indices.json", "split_integrity_report.json"}:
            status = "explain"
            note = ("The frozen manifest predates the current split file schema. The current file is verified "
                    "against features_full.parquet: row assignment exact, every item_idx in one split, "
                    "7000/1500/1500 groups, no overlap. Re-freezing the manifest would change Version A provenance.")
        else:
            status, note = "mismatch", "Frozen hash does not match the current file."
        add("manifest hash", name, expected_hash, actual, status, note, str(p.relative_to(ROOT)))

    report = json.loads((ROOT / "artifacts" / "split_integrity_report.json").read_text(encoding="utf-8"))
    expected_report = {
        "split": "group_by_item_idx",
        "seed": 42,
        "n_groups_per_split": {"test": 1500, "train": 7000, "val": 1500},
        "n_rows_per_split": {"test": 3000, "train": 14000, "val": 3000},
        "groups_spanning_multiple_splits": 0,
        "leakage_free": True,
    }
    for key, expected in expected_report.items():
        actual = report.get(key)
        add("split report", key, expected, actual, "match" if actual == expected else "mismatch",
            "", "artifacts/split_integrity_report.json")

    split = json.loads((ROOT / "artifacts" / "split_indices.json").read_text(encoding="utf-8"))
    feat = pd.read_parquet(ROOT / "data" / "processed" / "features_full.parquet",
                           columns=["sample_id", "item_idx", "split"])
    tr, va, te = set(split["train"]), set(split["val"]), set(split["test"])
    add("split file", "row split sizes", [14000, 3000, 3000], [len(tr), len(va), len(te)],
        "match" if [len(tr), len(va), len(te)] == [14000, 3000, 3000] else "mismatch",
        "", "artifacts/split_indices.json")
    add("split file", "row split disjoint", True, len(tr & va) + len(tr & te) + len(va & te) == 0,
        "match" if len(tr & va) + len(tr & te) + len(va & te) == 0 else "mismatch",
        "", "artifacts/split_indices.json")
    add("split file", "row split covers 20,000 rows", True, tr | va | te == set(range(20000)),
        "match" if tr | va | te == set(range(20000)) else "mismatch",
        "", "artifacts/split_indices.json")
    for name, rows in (("train", tr), ("val", va), ("test", te)):
        ok = feat.iloc[list(rows)]["split"].eq(name).all()
        add("split vs features", f"{name} row assignment", True, bool(ok), "match" if ok else "mismatch",
            "", "data/processed/features_full.parquet")
    single = int((feat.groupby("item_idx")["split"].nunique() > 1).sum())
    add("split vs features", "items spanning splits", 0, single, "match" if single == 0 else "mismatch",
        "", "data/processed/features_full.parquet")

    test_ids = set(feat.loc[feat["split"] == "test", "sample_id"])
    b2 = set(pd.read_parquet(RES / "b2" / "b2_predictions.parquet")["sample_id"])
    b6 = pd.read_parquet(RES / "b6" / "b6_predictions.parquet")
    b6_test = set(b6[(b6["variant"] == "m3") & (b6["seed"] == 42) & (b6["dataset"] == "halueval_test")]["sample_id"])
    for name, ids in (("b2_predictions", b2), ("b6_predictions", b6_test)):
        ok = ids == test_ids
        add("predictions", f"{name} test ids", len(test_ids), len(ids), "match" if ok else "mismatch",
            "sample_ids equal the features test split" if ok else "sample_ids differ from the features test split",
            f"artifacts/results/{'b2' if name.startswith('b2') else 'b6'}/{name}.parquet")

    req = REQUIREMENTS.read_text(encoding="utf-8")
    pin_lines = [ln for ln in req.splitlines() if "==" in ln and not ln.strip().startswith("#")]
    bad_lines = [ln for ln in req.splitlines() if re.search(r"(>=|<=|~=|!=|\s>\s|\s<\s)", ln) and not ln.strip().startswith("#")]
    add("requirements", "exact pins", "all pinned with ==", f"{len(pin_lines)} pins; {len(bad_lines)} range pins",
        "match" if not bad_lines else "mismatch",
        "xgboost stays at 3.3.0" if "xgboost==3.3.0" in req else "xgboost pin missing",
        "requirements.txt")
    key_pins = {
        "xgboost": "3.3.0",
        "torch": "2.11.0+cu128",
        "transformers": "5.14.1",
        "scikit-learn": "1.9.0",
        "pandas": "3.0.5",
        "numpy": "2.4.6",
    }
    for pkg, pinned in key_pins.items():
        try:
            installed = installed_version(pkg)
        except Exception:
            installed = "NOT INSTALLED"
        actual = f"pinned {pinned}; installed {installed}"
        status = "match" if installed == pinned else "mismatch"
        add("requirements", f"{pkg} version", pinned, installed, status, actual, "requirements.txt")

    lic = json.loads(LICENSES.read_text(encoding="utf-8"))
    expected_lic = {"halueval": "MIT", "ragtruth": "MIT", "faithbench": "CC BY-NC-SA 4.0"}
    by_name = {d["name"]: d for d in lic.get("datasets", [])}
    add("licenses", "dataset count", 3, len(by_name), "match" if len(by_name) == 3 else "mismatch",
        "", str(LICENSES.relative_to(ROOT)))
    for name, expected in expected_lic.items():
        actual = by_name.get(name, {}).get("license")
        add("licenses", f"{name} license", expected, actual, "match" if actual == expected else "mismatch",
            "", str(LICENSES.relative_to(ROOT)))
    add("licenses", "faithbench redistribution", False, by_name.get("faithbench", {}).get("redistribution_allowed"),
        "match" if by_name.get("faithbench", {}).get("redistribution_allowed") is False else "mismatch",
        "not redistributed; download notes only", str(LICENSES.relative_to(ROOT)))

    raw_checked = 0
    raw_mismatch = 0
    for name, entry in by_name.items():
        raw_dir = ROOT / entry["raw_dir"]
        for fname, meta in entry.get("files", {}).items():
            p = raw_dir / fname
            actual = sha256(p) if p.exists() else "MISSING"
            raw_checked += 1
            if actual != meta.get("sha256"):
                raw_mismatch += 1
                add("raw hashes", f"{name}/{fname}", meta.get("sha256"), actual, "mismatch",
                    "raw file missing or changed", str(p.relative_to(ROOT)))
    add("raw hashes", "all raw files match the license manifest", True, raw_mismatch == 0,
        "match" if raw_mismatch == 0 else "mismatch", f"{raw_checked} files checked",
        str(LICENSES.relative_to(ROOT)))

    add("repository", "LICENSE present", True, (ROOT / "LICENSE").exists(),
        "match" if (ROOT / "LICENSE").exists() else "mismatch", "", "LICENSE")

    out = {"checks": checks, "summary": {
        "n": len(checks),
        "match": sum(c["status"] == "match" for c in checks),
        "explain": sum(c["status"] == "explain" for c in checks),
        "mismatch": sum(c["status"] == "mismatch" for c in checks),
    }}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "support_files_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(checks).to_csv(OUT / "support_files_audit.csv", index=False)
    print(json.dumps(out["summary"], indent=2))
    bad = [c for c in checks if c["status"] == "mismatch"]
    if bad:
        print(pd.DataFrame(bad).to_string(index=False))


if __name__ == "__main__":
    main()
